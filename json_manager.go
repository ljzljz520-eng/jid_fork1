package jid

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"

	simplejson "github.com/bitly/go-simplejson"
	jmespath "github.com/jmespath/go-jmespath"
	jsoniter "github.com/json-iterator/go"
	"github.com/pkg/errors"
	"io"
)

var fastjson = jsoniter.ConfigCompatibleWithStandardLibrary

// Hot-path regexes, compiled once at package init. These run on every
// keystroke via isJMESPathQuery / getFilteredDataLegacy / getItem.
var (
	reJMESFuncCall      = regexp.MustCompile(`[a-z_]+\(`)
	reOpenBracketSuffix = regexp.MustCompile(`\[[0-9]*$`)
	reArrayIndex        = regexp.MustCompile(`\[([0-9]+)\]`)
)

// fdKey identifies one GetFilteredData computation. origin/originData are
// write-once (set in NewJsonManager), so results are pure functions of this key.
type fdKey struct {
	qs      string
	confirm bool
}

// fdResult caches one GetFilteredData result. The cached json is only ever
// read (Array/Map/GetIndex/Interface) and the slices are only reassigned,
// never mutated in place, by all callers — so returning the same values
// repeatedly is safe.
type fdResult struct {
	json       *simplejson.Json
	suggestion []string
	candidates []string
	err        error
	// node is the lossless result tree. Plain navigation reuses parsed
	// nodes (span-traceable); JMESPath-only results are synthetic.
	node *Node
}

type evalEntry struct {
	json *simplejson.Json
	err  error
}

// evalCacheMax bounds the per-keystroke JMESPath evaluation cache; when
// exceeded the whole map is dropped and rebuilt.
const evalCacheMax = 64

type JsonManager struct {
	current    *simplejson.Json
	origin     *simplejson.Json
	originData interface{}
	// tree is the lossless unified syntax tree of the original document
	// (token spans, number lexemes, ordered members with duplicates kept).
	tree *Node
	// mode selects lossless / canonical / standard output rendering.
	mode OutputStrategy
	// lastDiag holds precision/duplicate-key diagnostics for lastFD.
	lastDiag   Diagnostics
	suggestion *Suggestion
	// Single-entry memo for GetFilteredData/GetPretty. The TUI calls them
	// once per frame with an unchanged query while scrolling or cycling
	// candidates. Single-goroutine access only (termbox event loop).
	lastFDKey      fdKey
	lastFD         *fdResult // nil = no cached entry
	lastPretty     string    // MarshalIndent output for lastFDKey
	lastPrettyErr  error
	lastPrettyOK   bool
	lastPrettyMode OutputStrategy // strategy lastPretty was rendered with
	// evalCache memoizes evalJMESPath results: one keystroke can evaluate
	// the same expression several times (base expr, rewrites, suggestions).
	// Scoped to a single GetFilteredData computation — cleared on every
	// cache miss there — so result trees don't outlive the frame that
	// produced them.
	evalCache map[string]evalEntry
}

func NewJsonManager(reader io.Reader) (*JsonManager, error) {
	buf, err := io.ReadAll(reader)

	if err != nil {
		return nil, errors.Wrap(err, "invalid data")
	}

	j, err2 := simplejson.NewJson(buf)

	if err2 != nil {
		return nil, errors.Wrap(err2, "invalid json format")
	}

	var originData interface{}
	if err3 := json.Unmarshal(buf, &originData); err3 != nil {
		return nil, errors.Wrap(err3, "invalid json format")
	}

	// Lossless syntax tree (spans, number lexemes, ordered duplicate keys).
	tree, err4 := ParseDocument(string(buf))
	if err4 != nil {
		return nil, errors.Wrap(err4, "invalid json format")
	}

	jm := &JsonManager{
		origin:     j,
		current:    j,
		originData: originData,
		tree:       tree,
		mode:       StrategyLossless,
		suggestion: NewSuggestion(),
	}

	return jm, nil
}

// SetOutputStrategy changes the output rendering strategy (lossless /
// canonical / standard).
func (jm *JsonManager) SetOutputStrategy(mode OutputStrategy) {
	jm.mode = mode
}

// OutputStrategy returns the active output rendering strategy.
func (jm *JsonManager) OutputStrategy() OutputStrategy {
	return jm.mode
}

// Tree returns the parsed lossless document tree.
func (jm *JsonManager) Tree() *Node {
	return jm.tree
}

// LastDiagnostics returns precision/duplicate-key diagnostics collected for
// the most recent GetFilteredData result.
func (jm *JsonManager) LastDiagnostics() Diagnostics {
	return jm.lastDiag
}

// LastResultNode returns the lossless tree node backing the most recent
// GetFilteredData result, or nil when nothing has been queried yet.
func (jm *JsonManager) LastResultNode() *Node {
	if jm.lastFD == nil {
		return nil
	}
	return jm.lastFD.node
}

func (jm *JsonManager) Get(q QueryInterface, confirm bool) (string, []string, []string, error) {
	j, suggestion, candidates, _ := jm.GetFilteredData(q, confirm)

	if jm.lastFD != nil && jm.lastFD.node != nil {
		s, err := RenderNode(jm.lastFD.node, jm.mode, false)
		if err != nil {
			// DuplicateKeyError etc.: surface unchanged.
			return "", suggestion, candidates, err
		}
		return s, suggestion, candidates, nil
	}

	data, enc_err := fastjson.Marshal(j.Interface())
	if enc_err != nil {
		return "", []string{"", ""}, []string{"", ""}, errors.Wrap(enc_err, "failure json encode")
	}

	return string(data), suggestion, candidates, nil
}

func (jm *JsonManager) GetPretty(q QueryInterface, confirm bool) (string, []string, []string, error) {
	key := fdKey{q.StringGet(), confirm}
	_, suggestion, candidates, _ := jm.GetFilteredData(q, confirm)
	if jm.lastPrettyOK && jm.lastFDKey == key && jm.lastPrettyMode == jm.mode {
		if jm.lastPrettyErr != nil {
			return "", []string{"", ""}, []string{"", ""}, jm.lastPrettyErr
		}
		return jm.lastPretty, suggestion, candidates, nil
	}
	var s string
	var err error
	if jm.lastFD != nil && jm.lastFD.node != nil {
		s, err = RenderNode(jm.lastFD.node, jm.mode, true)
	} else {
		var b []byte
		b, err = fastjson.MarshalIndent(jm.lastFD.json.Interface(), "", "  ")
		s = string(b)
	}
	if err != nil {
		if jm.lastFD != nil && jm.lastFDKey == key {
			jm.lastPretty, jm.lastPrettyErr, jm.lastPrettyOK, jm.lastPrettyMode = "", err, true, jm.mode
		}
		return "", []string{"", ""}, []string{"", ""}, err
	}
	if jm.lastFD != nil && jm.lastFDKey == key {
		jm.lastPretty, jm.lastPrettyErr, jm.lastPrettyOK, jm.lastPrettyMode = s, nil, true, jm.mode
	}
	return s, suggestion, candidates, nil
}

// isJMESPathQuery returns true when the query contains JMESPath-specific syntax
// that goes beyond the simple dot/bracket path notation jid already supports.
// Specifically: pipe expressions, wildcard projections, filter expressions,
// function calls, multi-select hashes, or bare @ references.
func isJMESPathQuery(qs string) bool {
	inner := strings.TrimPrefix(qs, ".")
	// pipe: "|" (with or without surrounding spaces)
	if strings.Contains(inner, "|") {
		return true
	}
	// wildcard array projection [*] or .*
	if strings.Contains(inner, "[*]") || strings.Contains(inner, ".*") {
		return true
	}
	// filter expression [?
	if strings.Contains(inner, "[?") {
		return true
	}
	// function call: word(
	if reJMESFuncCall.MatchString(inner) {
		return true
	}
	// multi-select hash or bare @ reference
	if strings.ContainsAny(inner, "@{}") {
		return true
	}
	return false
}

// jmespathExprFromQuery converts jid's leading-dot query string to a JMESPath expression.
// "."              -> "@"
// ".foo.bar"       -> "foo.bar"
// ". | keys(@)"    -> "keys(@)"   (root + pipe → just the function)
// ".foo | keys(@)" -> "foo | keys(@)"
// ".[0]|keys(@)"   -> "[0]|keys(@)"
func jmespathExprFromQuery(qs string) string {
	expr := strings.TrimPrefix(qs, ".")
	if expr == "" {
		return "@"
	}
	// ". | func" becomes " | func" after TrimPrefix — strip the leading " | "
	if strings.HasPrefix(expr, " | ") {
		return expr[3:]
	}
	// ".|func" after TrimPrefix — strip leading "|"
	if strings.HasPrefix(expr, "|") {
		return strings.TrimPrefix(expr[1:], " ")
	}
	return expr
}

// evalJMESPath evaluates a JMESPath expression against the raw JSON data and
// returns the result as a *simplejson.Json.
func (jm *JsonManager) evalJMESPath(expr string) (*simplejson.Json, error) {
	if e, ok := jm.evalCache[expr]; ok {
		return e.json, e.err
	}
	j, err := jm.evalJMESPathUncached(expr)
	// Errors are cached too: evaluation is deterministic on the immutable
	// originData. Drop the whole map when it grows past the bound.
	if len(jm.evalCache) >= evalCacheMax {
		jm.evalCache = nil
	}
	if jm.evalCache == nil {
		jm.evalCache = make(map[string]evalEntry)
	}
	jm.evalCache[expr] = evalEntry{j, err}
	return j, err
}

func (jm *JsonManager) evalJMESPathUncached(expr string) (*simplejson.Json, error) {
	result, err := jmespath.Search(expr, jm.originData)
	if err != nil {
		return nil, err
	}
	b, err := json.Marshal(result)
	if err != nil {
		return nil, err
	}
	return simplejson.NewJson(b)
}

// evalBaseExpr evaluates expr like evalJMESPath but transparently rewrites
// wildcard-projection + numeric-index patterns (e.g. "foo[*].bar[0]") to
// pipe form ("foo[*].bar | [0]") when the direct evaluation returns an empty
// array. This matches the same fix applied in getFilteredDataJMESPath.
func (jm *JsonManager) evalBaseExpr(expr string) (*simplejson.Json, error) {
	result, err := jm.evalJMESPath(expr)
	// Apply wildcard+index rewrite both when eval errors (e.g. "foo[*].bar[0] | keys(@)"
	// where keys fails on []) and when it returns an empty array (e.g. "foo[*].bar[0]").
	needRewrite := err != nil
	if err == nil {
		if arr, arrErr := result.Array(); arrErr == nil && len(arr) == 0 {
			needRewrite = true
		}
	}
	if needRewrite {
		if m := reWildcardIndexed.FindStringSubmatch(expr); m != nil {
			pipeExpr := m[1] + " | [" + m[2] + "]" + m[3]
			if pipeResult, perr := jm.evalJMESPath(pipeExpr); perr == nil {
				return pipeResult, nil
			}
		}
	}
	return result, err
}

// reWildcardFieldTyping matches a JMESPath expression where the user is typing
// a field name after a wildcard projection, e.g. "foo[*].partial".
// Group 1: base expression (e.g. "foo[*]"), Group 2: partial field name (may be empty).
var reWildcardFieldTyping = regexp.MustCompile(`^(.*\[\*\])\.(\w*)$`)

// reWildcardIndexed matches a wildcard projection expression that contains a
// numeric index after the wildcard, e.g. "foo[*].bar[0]" or "foo[*].bar[0].name".
// In JMESPath, [N] within a projection applies to each element rather than the
// projected array.  The fix is to rewrite as "foo[*].bar | [0]" (or
// "foo[*].bar | [0].name") so [N] indexes the whole projected array.
// Group 1: expression before [N], Group 2: index digit(s), Group 3: path after [N].
var reWildcardIndexed = regexp.MustCompile(`^(.*\[\*\].*?)\[(\d+)\](.*)`)

// lastPipeIndex returns the index of the last `|` character in s,
// ignoring `|` inside filter expressions `[?...]`.
func lastPipeIndex(s string) int {
	return strings.LastIndex(s, "|")
}

// baseExprBeforePipe returns the JMESPath expression that precedes the last
// `|` (with or without surrounding spaces) in the query.
// Returns "@" if the base is the root (".").
// Returns "" if there is no pipe at all.
func baseExprBeforePipe(qs string) (string, bool) {
	inner := strings.TrimPrefix(qs, ".")
	idx := lastPipeIndex(inner)
	if idx < 0 {
		return "", false
	}
	base := strings.TrimRight(inner[:idx], " ")
	if base == "" {
		return "@", true // root
	}
	return base, true
}

// pipeSuffix returns everything after the last `|` in the query (spaces trimmed).
func pipeSuffix(qs string) string {
	inner := strings.TrimPrefix(qs, ".")
	if idx := lastPipeIndex(inner); idx >= 0 {
		return strings.TrimPrefix(inner[idx+1:], " ")
	}
	return ""
}

func (jm *JsonManager) GetFilteredData(q QueryInterface, confirm bool) (*simplejson.Json, []string, []string, error) {
	qs := q.StringGet()

	key := fdKey{qs, confirm}
	if jm.lastFD != nil && jm.lastFDKey == key {
		r := jm.lastFD
		// The strategy can change between frames; recompute diagnostics
		// even when the filtered result itself is cached.
		jm.lastDiag = CollectDiagnostics(r.node, jm.mode)
		return r.json, r.suggestion, r.candidates, r.err
	}

	// The query changed: release eval results kept for the previous one.
	// All evalJMESPath calls happen inside the computation below, so this
	// keeps the cache scoped to deduplicating within one filtering pass.
	jm.evalCache = nil

	var r fdResult
	if isJMESPathQuery(qs) {
		r.json, r.suggestion, r.candidates, r.err = jm.getFilteredDataJMESPath(qs, confirm)
		// Prefer a span-traceable tree result when the expression is plain
		// navigation; verify it matches the simplejson view exactly so the
		// suggestion fallbacks (typing modes, wildcard rewrites) stay
		// authoritative. Otherwise wrap the JMESPath output synthetically.
		r.node = jm.resolveResultNode(jmespathExprFromQuery(qs), r.json)
	} else {
		r.json, r.suggestion, r.candidates, r.err, r.node = jm.getFilteredDataLegacy(q, confirm)
	}
	jm.lastDiag = CollectDiagnostics(r.node, jm.mode)
	jm.lastFDKey, jm.lastFD, jm.lastPrettyOK = key, &r, false
	return r.json, r.suggestion, r.candidates, r.err
}

// resolveResultNode picks the lossless node backing a JMESPath result. It
// uses a referenced tree node only when (a) the expression belongs to the
// span-preserving navigation subset and (b) its canonical rendering equals
// what go-jmespath produced for the displayed simplejson view.
func (jm *JsonManager) resolveResultNode(expr string, sj *simplejson.Json) *Node {
	// A failed JMESPath evaluation yields no view; represent it as null
	// instead of dereferencing a nil simplejson result.
	if sj == nil {
		return syntheticNull()
	}
	if n, ok := evalPathOnTree(expr, jm.tree); ok {
		treeView, terr := marshalJSON(canonicalValue(n), false)
		if terr == nil {
			if b, jerr := fastjson.Marshal(sj.Interface()); jerr == nil && treeView == string(b) {
				return n
			}
		}
	}
	return nodeFromInterface(sj.Interface())
}

// isFunctionTypingMode returns true when the user appears to be mid-typing a
// JMESPath function name after a pipe: the suffix after " | " contains only
// identifier characters and no opening parenthesis yet.
func isFunctionTypingMode(qs string) bool {
	_, hasPipe := baseExprBeforePipe(qs)
	if !hasPipe {
		return false
	}
	suffix := pipeSuffix(qs)
	// Empty suffix means user just typed "|" and is about to enter a function name.
	// Return true as long as "(" hasn't appeared yet (function call not yet completed).
	return !strings.Contains(suffix, "(")
}

// ampFieldCandidates returns the base result with field candidates for &partial completion.
func (jm *JsonManager) ampFieldCandidates(baseExpr, partial string) (*simplejson.Json, []string, []string, error) {
	var baseResult *simplejson.Json
	if baseExpr == "@" {
		baseResult = jm.origin
	} else {
		var berr error
		baseResult, berr = jm.evalBaseExpr(baseExpr)
		if berr != nil {
			baseResult = jm.origin
		}
	}
	// For array base, use first element's keys
	el := baseResult
	if arr, arrErr := baseResult.Array(); arrErr == nil && len(arr) > 0 {
		el = baseResult.GetIndex(0)
	}
	fieldCandidates := jm.suggestion.GetCandidateKeys(el, partial)
	// If no candidates match (e.g. partial is placeholder text like "field"),
	// fall back to showing all keys.
	if len(fieldCandidates) == 0 {
		fieldCandidates = jm.suggestion.GetCandidateKeys(el, "")
	}
	// Don't emit a green inline hint: the cursor sits inside "&partial)" so any
	// suffix hint would appear after ")" and confuse the display. The candidate
	// list already shows all options.
	return baseResult, []string{"", ""}, fieldCandidates, nil
}

// ampFieldPartial detects when the user is typing a field name after `&` inside
// a function argument (e.g. "sort_by(@, &na" → partial="na", ok=true).
// Returns the partial identifier and true if the pattern is detected.
func ampFieldPartial(suffix string) (string, bool) {
	parenIdx := strings.Index(suffix, "(")
	if parenIdx < 0 {
		return "", false
	}
	ampIdx := strings.LastIndex(suffix, "&")
	if ampIdx < 0 || ampIdx < parenIdx {
		return "", false
	}
	partial := suffix[ampIdx+1:]
	// Strip trailing ) which may be present when the function template is already inserted
	partial = strings.TrimRight(partial, ")")
	// partial must contain only identifier characters (letters, digits, underscore)
	for _, ch := range partial {
		if ch != '_' && !('a' <= ch && ch <= 'z') && !('A' <= ch && ch <= 'Z') && !('0' <= ch && ch <= '9') {
			return "", false
		}
	}
	return partial, true
}

// getFilteredDataJMESPath handles queries that contain JMESPath-specific syntax.
func (jm *JsonManager) getFilteredDataJMESPath(qs string, confirm bool) (*simplejson.Json, []string, []string, error) {
	expr := jmespathExprFromQuery(qs)

	// If the user is mid-typing a function name after a pipe, don't evaluate
	// the (incomplete) expression. Show the base result and function suggestions.
	if isFunctionTypingMode(qs) {
		baseExpr, hasPipe := baseExprBeforePipe(qs)
		suffix := pipeSuffix(qs)
		if hasPipe {
			var baseResult *simplejson.Json
			if baseExpr == "@" {
				baseResult = jm.origin
			} else {
				var berr error
				baseResult, berr = jm.evalBaseExpr(baseExpr)
				if berr != nil {
					baseResult = jm.origin
				}
			}
			// When suffix is non-empty, prefer field candidates from the base object.
			if suffix != "" {
				fieldCandidates := jm.suggestion.GetCandidateKeys(baseResult, suffix)
				if len(fieldCandidates) > 0 {
					fieldSuggest := jm.suggestion.Get(baseResult, suffix)
					return baseResult, fieldSuggest, fieldCandidates, nil
				}
			}
			baseType := jm.suggestion.GetCurrentType(baseResult)
			fnCandidates := jm.suggestion.GetFunctionCandidatesFiltered(suffix, baseType)
			fnSuggest := jm.suggestion.GetFunctionSuggestionFiltered(suffix, baseType)
			return baseResult, fnSuggest, fnCandidates, nil
		}
	}

	// Try evaluating the full expression first.
	result, err := jm.evalJMESPath(expr)
	if err != nil {
		// The expression may contain a wildcard+index pattern whose [N] was
		// applied to each projected element instead of the array
		// (e.g. "foo[*].bar[0] | keys(@)"). Rewrite to pipe form and retry.
		if m := reWildcardIndexed.FindStringSubmatch(expr); m != nil {
			pipeExpr := m[1] + " | [" + m[2] + "]" + m[3]
			if pipeResult, perr := jm.evalJMESPath(pipeExpr); perr == nil {
				result = pipeResult
				err = nil
			}
		}
	}
	if err == nil {
		// For array results: wildcard projections ([*] or .*) produce an array of
		// objects; suggest element field keys so the user can type ".fieldname".
		if arr, arrErr := result.Array(); arrErr == nil {
			// When the query contains a wildcard ([*] or .*), the result is a
			// projection. If elements are objects, suggest their field keys so the
			// user can continue with ".fieldname" rather than "[0]".
			isWildcardExpr := strings.HasSuffix(expr, "[*]") || strings.HasSuffix(expr, ".*")
			if isWildcardExpr && len(arr) > 0 {
				firstEl := result.GetIndex(0)
				if candidateKeys := getCurrentKeys(firstEl); len(candidateKeys) > 0 {
					fieldSuggest := jm.suggestion.Get(firstEl, "")
					return result, fieldSuggest, candidateKeys, nil
				}
			}
			// Empty array may mean the user is typing a partial field name after a
			// wildcard projection (e.g. "game_indices[*].v"). JMESPath silently drops
			// null-projected values, so the result is []. Detect this and switch to
			// field-completion mode using the base wildcard expression.
			if len(arr) == 0 {
				// wildcard projection + numeric index (e.g. "foo[*].bar[0]" or
				// "foo[*].bar[0].name"): JMESPath applies [N] to each projected
				// element, not the array. Re-evaluate with pipe so [N] indexes
				// the whole projected array: "foo[*].bar | [0]" or "foo[*].bar | [0].name".
				if m := reWildcardIndexed.FindStringSubmatch(expr); m != nil {
					pipeExpr := m[1] + " | [" + m[2] + "]" + m[3]
					if pipeResult, perr := jm.evalJMESPath(pipeExpr); perr == nil {
						if candidateKeys := getCurrentKeys(pipeResult); len(candidateKeys) > 0 {
							fieldSuggest := jm.suggestion.Get(pipeResult, "")
							return pipeResult, fieldSuggest, candidateKeys, nil
						}
						pSuggest := jm.suggestion.Get(pipeResult, "")
						return pipeResult, pSuggest, []string{}, nil
					}
				}
				if m := reWildcardFieldTyping.FindStringSubmatch(expr); m != nil {
					baseExpr, partial := m[1], m[2]
					if baseResult, berr := jm.evalBaseExpr(baseExpr); berr == nil {
						if baseArr, bArrErr := baseResult.Array(); bArrErr == nil && len(baseArr) > 0 {
							firstEl := baseResult.GetIndex(0)
							fieldCandidates := jm.suggestion.GetCandidateKeys(firstEl, partial)
							// Only switch to field-completion mode if there are matching
							// candidates. If the partial doesn't match any field, fall
							// through and show the actual empty-array result.
							if len(fieldCandidates) > 0 || partial == "" {
								fieldSuggest := jm.suggestion.Get(firstEl, partial)
								return baseResult, fieldSuggest, fieldCandidates, nil
							}
						}
					}
				}
				// Empty array with no field-completion match: show result without
				// an index suggestion (appending [0] to an empty projection misleads).
				return result, []string{"", ""}, []string{}, nil
			}
			suggest := jm.suggestion.Get(result, "")
			return result, suggest, []string{}, nil
		}
		// Null result: may be &partial with a non-existent field (e.g. max_by(@, &field)).
		// Detect &partial pattern and show field candidates instead.
		// Skip this fallback when confirm=true so that a confirmed expression that
		// genuinely returns null (e.g. max_by(@, &missing)) produces null output.
		if result.Interface() == nil && !confirm {
			if baseExpr2, hasPipe2 := baseExprBeforePipe(qs); hasPipe2 {
				suffix2 := pipeSuffix(qs)
				if partial, ok := ampFieldPartial(suffix2); ok {
					return jm.ampFieldCandidates(baseExpr2, partial)
				}
			}
		}
		// Map (object) result: suggest field keys so the user can keep digging.
		if candidateKeys := getCurrentKeys(result); len(candidateKeys) > 0 {
			fieldSuggest := jm.suggestion.Get(result, "")
			return result, fieldSuggest, candidateKeys, nil
		}
		return result, []string{"", ""}, []string{}, nil
	}

	// Expression is incomplete (parse error).
	// When confirming (user pressed Enter), do not fall back to base-result recovery;
	// return the error so the caller can display it correctly.
	if confirm {
		return jm.origin, []string{"", ""}, []string{}, err
	}

	// Check whether the user is typing after a pipe.
	baseExpr, hasPipe := baseExprBeforePipe(qs)
	suffix := pipeSuffix(qs)

	if hasPipe {
		// Suffix ends with ".": user typed a dot to start field navigation after a
		// complete pipe expression (e.g. ". | to_array(@)[0]."). Try evaluating the
		// expression without the trailing dot and offer its field candidates.
		if strings.HasSuffix(suffix, ".") {
			exprWithoutDot := strings.TrimSuffix(expr, ".")
			if tempResult, berr := jm.evalJMESPath(exprWithoutDot); berr == nil {
				if candidateKeys := getCurrentKeys(tempResult); len(candidateKeys) > 0 {
					fieldSuggest := jm.suggestion.Get(tempResult, "")
					return tempResult, fieldSuggest, candidateKeys, nil
				}
				if arr2, arrErr2 := tempResult.Array(); arrErr2 == nil && len(arr2) > 0 {
					firstEl := tempResult.GetIndex(0)
					if candidateKeys := getCurrentKeys(firstEl); len(candidateKeys) > 0 {
						fieldSuggest := jm.suggestion.Get(firstEl, "")
						return tempResult, fieldSuggest, candidateKeys, nil
					}
				}
			}
		}

		// Show the result of the base expression while the user types the function.
		var baseResult *simplejson.Json
		if baseExpr == "@" {
			baseResult = jm.origin
		} else {
			var berr error
			baseResult, berr = jm.evalBaseExpr(baseExpr)
			if berr != nil {
				baseResult = jm.origin
			}
		}
		// Detect &partial pattern in function argument (e.g. "sort_by(@, &ba").
		// Skip when confirm=true so that a confirmed query returns its actual result.
		if !confirm {
			if partial, ok := ampFieldPartial(suffix); ok {
				return jm.ampFieldCandidates(baseExpr, partial)
			}
		}

		// Provide function-name candidates matching the typed suffix.
		baseType := jm.suggestion.GetCurrentType(baseResult)
		fnCandidates := jm.suggestion.GetFunctionCandidatesFiltered(suffix, baseType)
		fnSuggest := jm.suggestion.GetFunctionSuggestionFiltered(suffix, baseType)
		return baseResult, fnSuggest, fnCandidates, nil
	}

	// No pipe. Check if the user typed a trailing "." to start field navigation
	// after a wildcard or index expression (e.g. "game_indices[*].").
	// Strip the dot, re-evaluate, and offer field candidates from the result.
	if strings.HasSuffix(expr, ".") {
		baseExpr := strings.TrimSuffix(expr, ".")
		if baseResult, berr := jm.evalBaseExpr(baseExpr); berr == nil {
			// Array-of-objects: suggest element field keys (wildcard projection).
			if arr, arrErr := baseResult.Array(); arrErr == nil && len(arr) > 0 {
				firstEl := baseResult.GetIndex(0)
				if candidateKeys := getCurrentKeys(firstEl); len(candidateKeys) > 0 {
					fieldSuggest := jm.suggestion.Get(firstEl, "")
					return baseResult, fieldSuggest, candidateKeys, nil
				}
			}
			// Plain object: suggest its keys.
			if candidateKeys := getCurrentKeys(baseResult); len(candidateKeys) > 0 {
				fieldSuggest := jm.suggestion.Get(baseResult, "")
				return baseResult, fieldSuggest, candidateKeys, nil
			}
		}
	}

	// Expression is partially typed but invalid – fall back to origin.
	return jm.origin, []string{"", ""}, []string{}, nil
}

// getFilteredDataLegacy is the original keyword-traversal logic, unchanged
// in its simplejson behaviour. The returned *Node tracks the same traversal
// on the lossless tree so the result keeps its source span; missing steps
// map to synthetic null nodes.
func (jm *JsonManager) getFilteredDataLegacy(q QueryInterface, confirm bool) (*simplejson.Json, []string, []string, error, *Node) {
	json := jm.origin
	node := jm.tree

	lastKeyword := q.StringGetLastKeyword()
	keywords := q.StringGetKeywords()

	idx := 0
	if l := len(keywords); l == 0 {
		return json, []string{"", ""}, []string{}, nil, node
	} else if l > 0 {
		idx = l - 1
	}
	for _, keyword := range keywords[0:idx] {
		json, _ = getItem(json, keyword)
		node = nodeStep(node, keyword)
	}
	suggest := jm.suggestion.Get(json, lastKeyword)
	candidateKeys := jm.suggestion.GetCandidateKeys(json, lastKeyword)
	// hash
	if len(reOpenBracketSuffix.FindString(lastKeyword)) < 1 {
		candidateNum := len(candidateKeys)
		if j, exist := getItem(json, lastKeyword); exist && (confirm || candidateNum == 1) {
			json = j
			node = nodeStep(node, lastKeyword)
			candidateKeys = []string{}
			if _, err := json.Array(); err == nil {
				suggest = jm.suggestion.Get(json, "")
			} else {
				suggest = []string{"", ""}
			}
		} else if candidateNum < 1 {
			// j is the getItem result regardless of the confirm condition
			// (an out-of-range/missing value encodes as null); nodeStep
			// mirrors that with the referenced node or a synthetic null.
			json = j
			node = nodeStep(node, lastKeyword)
			suggest = jm.suggestion.Get(json, "")
		}
	}
	return json, suggest, candidateKeys, nil, node
}

func (jm *JsonManager) GetCandidateKeys(q QueryInterface) []string {
	return jm.suggestion.GetCandidateKeys(jm.current, q.StringGetLastKeyword())
}

func getItem(json *simplejson.Json, s string) (*simplejson.Json, bool) {
	var result *simplejson.Json
	var exist bool

	matches := reArrayIndex.FindStringSubmatch(s)

	if s == "" {
		return json, false
	}

	// Query include [
	if len(matches) > 0 {
		index, _ := strconv.Atoi(matches[1])
		if a, err := json.Array(); err != nil {
			exist = false
		} else if len(a) < index {
			exist = false
		}
		result = json.GetIndex(index)
	} else {
		result, exist = json.CheckGet(s)
		if result == nil {
			result = &simplejson.Json{}
		}
	}
	return result, exist
}

func isEmptyJson(j *simplejson.Json) bool {
	switch j.Interface().(type) {
	case nil:
		return true
	default:
		return false
	}
}
