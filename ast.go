package jid

// Unified, lossless JSON syntax tree with byte/rune token spans.
//
// Every value produced by the parser is a *Node carrying the exact Span of
// its token(s) in the original document. Numbers keep their original lexeme
// (so integers beyond 2^53 and high-precision decimals round-trip byte for
// byte) and can be evaluated on demand as *big.Int / *big.Rat / *big.Float.
// Objects are ordered member lists and keep duplicate keys.
//
// Nodes produced by query evaluation (JMESPath projections, function
// results) are marked Synthetic: they have no source span. Navigation over
// plain path expressions (fields, indexes, [*] / .* projections, pipes of
// those) reuses parsed nodes, so a query result can be traced back to the
// exact byte/rune range it came from.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/pkg/errors"
)

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

// Span is the exact location of a token in the original JSON source.
// Byte offsets are byte indices (half-open); rune offsets count Unicode
// code points. Line/Col are 1-based, columns are counted in runes.
type Span struct {
	ByteStart int `json:"byte_start"`
	ByteEnd   int `json:"byte_end"`
	RuneStart int `json:"rune_start"`
	RuneEnd   int `json:"rune_end"`
	Line      int `json:"line"`
	Col       int `json:"col"`
}

// SourceText returns the verbatim source text of the span and true when the
// node originates from parsed input.
func (n *Node) SourceText() (string, bool) {
	if n == nil || n.doc == nil {
		return "", false
	}
	return n.doc.src[n.Span.ByteStart:n.Span.ByteEnd], true
}

// ---------------------------------------------------------------------------
// Output strategies
// ---------------------------------------------------------------------------

// OutputStrategy selects how query results are serialized.
type OutputStrategy string

const (
	// StrategyLossless (default) renders referenced subtrees from their
	// original source text, keeps every duplicate key in document order and
	// never reinterprets a number lexeme.
	StrategyLossless OutputStrategy = "lossless"
	// StrategyCanonical renders through float64 / map[string]interface{}
	// exactly like encoding/json: map keys are sorted and duplicate keys
	// collapse (last wins). Numbers beyond float64 precision may change.
	StrategyCanonical OutputStrategy = "canonical"
	// StrategyStandard behaves like canonical output but rejects any object
	// that contains duplicate keys instead of silently collapsing them.
	StrategyStandard OutputStrategy = "standard"
)

// ParseOutputStrategy parses a strategy name. Empty string means lossless.
func ParseOutputStrategy(s string) (OutputStrategy, error) {
	switch OutputStrategy(strings.ToLower(strings.TrimSpace(s))) {
	case "", StrategyLossless:
		return StrategyLossless, nil
	case StrategyCanonical:
		return StrategyCanonical, nil
	case StrategyStandard, "strict":
		return StrategyStandard, nil
	default:
		return "", errors.Errorf("unknown output strategy %q (want one of: lossless, canonical, standard)", s)
	}
}

// DuplicateKeyError is returned by the "standard" strategy when a subtree
// selected for output contains an object with duplicate keys.
type DuplicateKeyError struct {
	Key  string
	Span Span
}

func (e *DuplicateKeyError) Error() string {
	return fmt.Sprintf(
		"duplicate key %q at line %d column %d (byte offset %d): strategy \"standard\" rejects duplicate keys",
		e.Key, e.Span.Line, e.Span.Col, e.Span.ByteStart)
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

// NodeType enumerates JSON value types.
type NodeType int

const (
	NNull NodeType = iota
	NBool
	NString
	NNumber
	NObject
	NArray
)

// Member is one object member. Members are stored in document order and
// duplicates are retained.
type Member struct {
	Key   *Node // NString node; its Span covers the quoted key token
	Value *Node
}

// sourceDoc is shared by all nodes parsed from the same input.
type sourceDoc struct {
	src string
}

// Node is one value in the unified syntax tree.
type Node struct {
	Type NodeType
	Span Span

	// Scalars.
	Str  string       // decoded value for NString
	Bool bool         // value for NBool
	Num  *NumberValue // parsed number for NNumber originating from source

	// Composite containers.
	Members []Member // NObject, ordered, duplicates kept
	Elems   []*Node  // NArray, ordered

	// Provenance.
	doc       *sourceDoc // non-nil for nodes read from the original document
	root      bool       // this node is the document root
	Synthetic bool       // produced by query evaluation, no source span

	// f64 holds the float64 value for synthetic number nodes (Num == nil).
	f64 float64
}

// IsNull reports whether the node is a JSON null.
func (n *Node) IsNull() bool { return n != nil && n.Type == NNull }

// TypeName returns the JMESPath/JSON type name of the node.
func (n *Node) TypeName() string {
	switch n.Type {
	case NNull:
		return "null"
	case NBool:
		return "boolean"
	case NString:
		return "string"
	case NNumber:
		return "number"
	case NObject:
		return "object"
	case NArray:
		return "array"
	}
	return "unknown"
}

// syntheticNull returns a span-less null node.
func syntheticNull() *Node { return &Node{Type: NNull, Synthetic: true} }

// newSyntheticNumber wraps a float64 produced by JMESPath evaluation.
func newSyntheticNumber(f float64) *Node {
	return &Node{Type: NNumber, Synthetic: true, f64: f}
}

func newSyntheticArray(elems []*Node) *Node {
	return &Node{Type: NArray, Synthetic: true, Elems: elems}
}

// ---------------------------------------------------------------------------
// NumberValue
// ---------------------------------------------------------------------------

// NumberValue keeps the original JSON number lexeme and evaluates it exactly
// on demand.
type NumberValue struct {
	Lexeme string

	isInt    bool
	parsed   bool
	rat      *big.Rat // exact decimal value
	intVal   *big.Int // set when isInt
	scale    int      // decimal fractional digits implied by the lexeme
	parseOK  bool
	losesF64 bool
	over2p53 bool
}

// safe-integer bounds (2^53), beyond which float64 integers lose units.
var (
	maxSafeInt = new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 53), big.NewInt(1))
	minSafeInt = new(big.Int).Neg(maxSafeInt)
)

// parseNumberLexeme validates the JSON number grammar and converts the
// lexeme into an exact rational (num/den before reduction). scale is the
// number of fractional decimal digits after applying the exponent
// (clamped at 0).
func parseNumberLexeme(lit string) (num *big.Int, den *big.Int, scale int, ok bool) {
	s := lit
	if s == "" {
		return nil, nil, 0, false
	}
	if s[0] == '-' {
		s = s[1:]
	}
	if s == "" {
		return nil, nil, 0, false
	}
	intPart := s
	fracPart := ""
	expPart := ""
	if i := strings.IndexAny(s, ".eE"); i >= 0 {
		intPart = s[:i]
		rest := s[i:]
		if rest[0] == '.' {
			rest = rest[1:]
			j := 0
			for j < len(rest) && rest[j] >= '0' && rest[j] <= '9' {
				j++
			}
			if j == 0 {
				return nil, nil, 0, false // "." must be followed by a digit
			}
			fracPart = rest[:j]
			rest = rest[j:]
		}
		if len(rest) > 0 && (rest[0] == 'e' || rest[0] == 'E') {
			expPart = rest[1:]
			rest = rest[len(rest):]
		}
		if rest != "" {
			return nil, nil, 0, false
		}
	}
	// Integer part: "0" or [1-9][0-9]*.
	if intPart == "" || (len(intPart) > 1 && intPart[0] == '0') {
		return nil, nil, 0, false
	}
	for i := 0; i < len(intPart); i++ {
		if intPart[i] < '0' || intPart[i] > '9' {
			return nil, nil, 0, false
		}
	}
	exp := 0
	if expPart != "" {
		es := expPart
		if es[0] == '+' || es[0] == '-' {
			es = es[1:]
		}
		if es == "" {
			return nil, nil, 0, false
		}
		v, err := strconv.Atoi(expPart)
		if err != nil {
			return nil, nil, 0, false
		}
		exp = v
	}
	digits := intPart + fracPart
	num, good := new(big.Int).SetString(digits, 10)
	if !good {
		return nil, nil, 0, false
	}
	if lit[0] == '-' {
		num.Neg(num)
	}
	power := exp - len(fracPart)
	if power >= 0 {
		num.Mul(num, new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(power)), nil))
		den = big.NewInt(1)
	} else {
		den = new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(-power)), nil)
	}
	if d := len(fracPart) - exp; d > 0 {
		scale = d
	}
	return num, den, scale, true
}

func (v *NumberValue) ensureParsed() {
	if v.parsed {
		return
	}
	v.parsed = true
	num, den, scale, ok := parseNumberLexeme(v.Lexeme)
	if !ok {
		return
	}
	v.parseOK = true
	v.scale = scale
	v.rat = new(big.Rat).SetFrac(num, den)
	// A number is an integer when its exact value is a whole number,
	// regardless of lexical form (e.g. 1E3 and 1.00 evaluate to 1000/1).
	if v.rat.IsInt() {
		v.isInt = true
		v.intVal = new(big.Int).Set(v.rat.Num())
		if v.intVal.Cmp(maxSafeInt) > 0 || v.intVal.Cmp(minSafeInt) < 0 {
			v.over2p53 = true
		}
	}
	// Float64 round-trip check: the lexeme survives a float64 conversion
	// only when float64(lexeme) converts back to the exact same rational.
	f, _ := v.rat.Float64()
	if math.IsInf(f, 0) {
		v.losesF64 = true
	} else {
		back := new(big.Rat).SetFloat64(f)
		if back == nil || back.Cmp(v.rat) != 0 {
			v.losesF64 = true
		}
	}
}

// Valid reports whether the lexeme is a well-formed JSON number.
func (v *NumberValue) Valid() bool { v.ensureParsed(); return v.parseOK }

// IsInteger reports whether the exact value is a whole number (e.g. 1E3
// and 1.00 count; the lexeme itself is always kept in Lexeme).
func (v *NumberValue) IsInteger() bool { v.ensureParsed(); return v.parseOK && v.isInt }

// BigInt returns the exact integer value and true for integer literals.
func (v *NumberValue) BigInt() (*big.Int, bool) {
	v.ensureParsed()
	if !v.parseOK || !v.isInt {
		return nil, false
	}
	return new(big.Int).Set(v.intVal), true
}

// BigRat returns the exact rational value (terminating decimals included)
// and true.
func (v *NumberValue) BigRat() (*big.Rat, bool) {
	v.ensureParsed()
	if !v.parseOK {
		return nil, false
	}
	return new(big.Rat).Set(v.rat), true
}

// Decimal evaluates the lexeme as an arbitrary-precision decimal floating
// point. Precision is sized generously from the significant-digit count of
// the lexeme (minimum 256 bits); BigRat remains the exactly-representable
// alternative.
func (v *NumberValue) Decimal() (*big.Float, bool) {
	v.ensureParsed()
	if !v.parseOK {
		return nil, false
	}
	sig := v.Lexeme
	if i := strings.IndexAny(sig, "eE"); i >= 0 {
		sig = sig[:i]
	}
	sig = strings.NewReplacer(".", "", "-", "").Replace(sig)
	digits := len(sig)
	prec := uint(256)
	if need := uint(digits*4 + 64); need > prec {
		prec = need
	}
	f := new(big.Float).SetPrec(prec)
	f.SetRat(v.rat)
	return f, true
}

// Scale returns the number of fractional decimal digits implied by the
// lexeme (after applying its exponent), e.g. "1.20" -> 2, "1.2e2" -> 0.
func (v *NumberValue) Scale() int { v.ensureParsed(); return v.scale }

// Float64 returns the float64 interpretation (as encoding/json would use).
func (v *NumberValue) Float64() (float64, bool) {
	v.ensureParsed()
	if !v.parseOK {
		return 0, false
	}
	f, _ := v.rat.Float64()
	return f, true
}

// LosesFloat64Precision reports whether the lexeme cannot be represented
// exactly as a float64 (integer beyond 2^53 or a non-dyadic decimal).
func (v *NumberValue) LosesFloat64Precision() bool {
	v.ensureParsed()
	return v.parseOK && v.losesF64
}

// BeyondSafeInteger reports whether the integer lexeme has magnitude > 2^53.
func (v *NumberValue) BeyondSafeInteger() bool {
	v.ensureParsed()
	return v.parseOK && v.isInt && v.over2p53
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

type jsonParser struct {
	doc   *sourceDoc
	pos   int // byte offset
	runes int // rune offset
	line  int // 1-based
	col   int // 1-based rune column
}

type posMark struct {
	pos, runes, line, col int
}

func (p *jsonParser) mark() posMark {
	return posMark{p.pos, p.runes, p.line, p.col}
}

// closeSpan records the span from m to the current position and advances
// rune/column counters. JSON string tokens never contain raw newlines, so
// counting runes of the consumed slice is sufficient here.
func (p *jsonParser) closeSpan(m posMark) Span {
	text := p.doc.src[m.pos:p.pos]
	rc := utf8.RuneCountInString(text)
	sp := Span{
		ByteStart: m.pos,
		ByteEnd:   p.pos,
		RuneStart: m.runes,
		RuneEnd:   m.runes + rc,
		Line:      m.line,
		Col:       m.col,
	}
	p.runes += rc
	p.col += rc
	return sp
}

// consumePunct advances past one ASCII structural character ({, }, [, ],
// :, ,), updating the rune/column counters so later token spans carry
// correct absolute rune offsets and columns.
func (p *jsonParser) consumePunct() {
	p.pos++
	p.runes++
	p.col++
}

func (p *jsonParser) skipWS() error {
	for p.pos < len(p.doc.src) {
		r, size := utf8.DecodeRuneInString(p.doc.src[p.pos:])
		if r != ' ' && r != '\t' && r != '\n' && r != '\r' {
			return nil
		}
		if r == '\n' {
			p.line++
			p.col = 1
		} else {
			p.col++
		}
		p.runes++
		p.pos += size
	}
	return nil
}

func (p *jsonParser) errorf(m posMark, format string, args ...interface{}) error {
	return errors.Errorf("invalid json format at line %d column %d: %s",
		m.line, m.col, fmt.Sprintf(format, args...))
}

// ParseDocument parses src into a lossless syntax tree.
func ParseDocument(src string) (*Node, error) {
	p := &jsonParser{doc: &sourceDoc{src: src}, line: 1, col: 1}
	if err := p.skipWS(); err != nil {
		return nil, err
	}
	n, err := p.parseValue()
	if err != nil {
		return nil, err
	}
	if err := p.skipWS(); err != nil {
		return nil, err
	}
	if p.pos != len(src) {
		m := p.mark()
		return nil, p.errorf(m, "unexpected trailing data %q", preview(src[p.pos:]))
	}
	n.root = true
	return n, nil
}

func preview(s string) string {
	if len(s) > 16 {
		return s[:16] + "..."
	}
	return s
}

func (p *jsonParser) parseValue() (*Node, error) {
	if err := p.skipWS(); err != nil {
		return nil, err
	}
	m := p.mark()
	if p.pos >= len(p.doc.src) {
		return nil, p.errorf(m, "unexpected end of input")
	}
	var n *Node
	var err error
	switch p.doc.src[p.pos] {
	case '{':
		n, err = p.parseObject()
	case '[':
		n, err = p.parseArray()
	case '"':
		n, err = p.parseStringNode()
	case 't':
		n, err = p.parseLiteral("true", true)
	case 'f':
		n, err = p.parseLiteral("false", false)
	case 'n':
		n, err = p.parseNull()
	default:
		n, err = p.parseNumberNode()
	}
	if err != nil {
		return nil, err
	}
	n.Span = p.closeSpan(m)
	return n, nil
}

func (p *jsonParser) parseLiteral(lit string, b bool) (*Node, error) {
	m := p.mark()
	if !strings.HasPrefix(p.doc.src[p.pos:], lit) {
		return nil, p.errorf(m, "invalid literal")
	}
	p.pos += len(lit)
	return &Node{Type: NBool, Bool: b, doc: p.doc}, nil
}

func (p *jsonParser) parseNull() (*Node, error) {
	m := p.mark()
	if !strings.HasPrefix(p.doc.src[p.pos:], "null") {
		return nil, p.errorf(m, "invalid literal")
	}
	p.pos += 4
	return &Node{Type: NNull, doc: p.doc}, nil
}

func (p *jsonParser) parseNumberNode() (*Node, error) {
	m := p.mark()
	start := p.pos
	for p.pos < len(p.doc.src) {
		c := p.doc.src[p.pos]
		if c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' || (c >= '0' && c <= '9') {
			p.pos++
			continue
		}
		break
	}
	lit := p.doc.src[start:p.pos]
	nv := &NumberValue{Lexeme: lit}
	nv.ensureParsed()
	if !nv.parseOK {
		return nil, p.errorf(m, "invalid number %q", lit)
	}
	return &Node{Type: NNumber, Num: nv, doc: p.doc}, nil
}

// scanString locates the closing quote of the string starting at p.pos
// (which must point at '"'), validates escapes and returns the end offset
// (exclusive). Raw control characters are rejected.
func (p *jsonParser) scanString() (int, error) {
	m := p.mark()
	i := p.pos + 1
	for i < len(p.doc.src) {
		c := p.doc.src[i]
		if c == '"' {
			return i + 1, nil
		}
		if c == '\\' {
			i++
			if i >= len(p.doc.src) {
				break
			}
			esc := p.doc.src[i]
			switch esc {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				i++
			case 'u':
				if i+4 >= len(p.doc.src) {
					return 0, p.errorf(m, "invalid unicode escape")
				}
				for j := 1; j <= 4; j++ {
					h := p.doc.src[i+j]
					if !((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F')) {
						return 0, p.errorf(m, "invalid unicode escape")
					}
				}
				i += 5
			default:
				return 0, p.errorf(m, "invalid escape %q", string(esc))
			}
			continue
		}
		if c < 0x20 {
			return 0, p.errorf(m, "unescaped control character in string")
		}
		i++
	}
	return 0, p.errorf(m, "unterminated string")
}

func (p *jsonParser) parseStringNode() (*Node, error) {
	m := p.mark()
	end, err := p.scanString()
	if err != nil {
		return nil, err
	}
	raw := p.doc.src[p.pos:end]
	v, err := strconv.Unquote(raw)
	if err != nil {
		return nil, p.errorf(m, "invalid string: %s", err.Error())
	}
	p.pos = end
	return &Node{Type: NString, Str: v, doc: p.doc}, nil
}

func (p *jsonParser) parseObject() (*Node, error) {
	n := &Node{Type: NObject, doc: p.doc}
	p.consumePunct() // {
	if err := p.skipWS(); err != nil {
		return nil, err
	}
	if p.pos >= len(p.doc.src) {
		return nil, p.errorf(p.mark(), "unterminated object")
	}
	if p.doc.src[p.pos] == '}' {
		p.consumePunct()
		return n, nil
	}
	for {
		if err := p.skipWS(); err != nil {
			return nil, err
		}
		if p.pos >= len(p.doc.src) || p.doc.src[p.pos] != '"' {
			return nil, p.errorf(p.mark(), "expected object key string")
		}
		keyMark := p.mark()
		key, err := p.parseStringNode()
		if err != nil {
			return nil, err
		}
		key.Span = p.closeSpan(keyMark)
		if err := p.skipWS(); err != nil {
			return nil, err
		}
		if p.pos >= len(p.doc.src) || p.doc.src[p.pos] != ':' {
			return nil, p.errorf(p.mark(), "expected ':' after object key")
		}
		p.consumePunct()
		val, err := p.parseValue()
		if err != nil {
			return nil, err
		}
		n.Members = append(n.Members, Member{Key: key, Value: val})
		if err := p.skipWS(); err != nil {
			return nil, err
		}
		if p.pos >= len(p.doc.src) {
			return nil, p.errorf(p.mark(), "unterminated object")
		}
		switch p.doc.src[p.pos] {
		case ',':
			p.consumePunct()
			continue
		case '}':
			p.consumePunct()
			return n, nil
		default:
			return nil, p.errorf(p.mark(), "expected ',' or '}'")
		}
	}
}

func (p *jsonParser) parseArray() (*Node, error) {
	n := &Node{Type: NArray, doc: p.doc}
	p.consumePunct() // [
	if err := p.skipWS(); err != nil {
		return nil, err
	}
	if p.pos >= len(p.doc.src) {
		return nil, p.errorf(p.mark(), "unterminated array")
	}
	if p.doc.src[p.pos] == ']' {
		p.consumePunct()
		return n, nil
	}
	for {
		val, err := p.parseValue()
		if err != nil {
			return nil, err
		}
		n.Elems = append(n.Elems, val)
		if err := p.skipWS(); err != nil {
			return nil, err
		}
		if p.pos >= len(p.doc.src) {
			return nil, p.errorf(p.mark(), "unterminated array")
		}
		switch p.doc.src[p.pos] {
		case ',':
			p.consumePunct()
			if err := p.skipWS(); err != nil {
				return nil, err
			}
			continue
		case ']':
			p.consumePunct()
			return n, nil
		default:
			return nil, p.errorf(p.mark(), "expected ',' or ']'")
		}
	}
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

// canonicalValue converts a node the same way encoding/json decodes into
// interface{}: numbers become float64 and objects become last-wins maps.
func canonicalValue(n *Node) interface{} {
	switch n.Type {
	case NNull:
		return nil
	case NBool:
		return n.Bool
	case NString:
		return n.Str
	case NNumber:
		if n.Num != nil {
			f, _ := n.Num.Float64()
			return f
		}
		return n.f64
	case NObject:
		m := make(map[string]interface{}, len(n.Members))
		for _, mem := range n.Members {
			m[mem.Key.Str] = canonicalValue(mem.Value) // last wins
		}
		return m
	case NArray:
		a := make([]interface{}, len(n.Elems))
		for i, el := range n.Elems {
			a[i] = canonicalValue(el)
		}
		return a
	}
	return nil
}

// nodeFromInterface builds a synthetic tree from decoded JMESPath output.
func nodeFromInterface(v interface{}) *Node {
	switch t := v.(type) {
	case nil:
		return syntheticNull()
	case bool:
		return &Node{Type: NBool, Bool: t, Synthetic: true}
	case string:
		return &Node{Type: NString, Str: t, Synthetic: true}
	case float64:
		return newSyntheticNumber(t)
	case json.Number:
		// JMESPath currently decodes with float64; keep a defensive path.
		nv := &NumberValue{Lexeme: string(t)}
		return &Node{Type: NNumber, Num: nv, Synthetic: true}
	case map[string]interface{}:
		n := &Node{Type: NObject, Synthetic: true}
		for k, val := range t {
			n.Members = append(n.Members, Member{
				Key:   &Node{Type: NString, Str: k, Synthetic: true},
				Value: nodeFromInterface(val),
			})
		}
		return n
	case []interface{}:
		n := &Node{Type: NArray, Synthetic: true, Elems: make([]*Node, len(t))}
		for i, val := range t {
			n.Elems[i] = nodeFromInterface(val)
		}
		return n
	default:
		return syntheticNull()
	}
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// marshalJSON renders v with standard encoding/json semantics.
func marshalJSON(v interface{}, pretty bool) (string, error) {
	var b []byte
	var err error
	if pretty {
		b, err = json.MarshalIndent(v, "", "  ")
	} else {
		b, err = json.Marshal(v)
	}
	if err != nil {
		return "", errors.Wrap(err, "failure json encode")
	}
	return string(b), nil
}

// quoteString JSON-encodes a decoded string without HTML-escaping: the
// lossless renderer must not rewrite '<', '>' or '&' found in the source.
func quoteString(s string) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(s)
	return strings.TrimRight(buf.String(), "\n")
}

// renderLossless re-renders a node preserving number lexemes, member order
// and duplicate keys. Referenced scalars/containers are rebuilt value by
// value; compact rendering of referenced nodes uses the verbatim span text
// instead (see RenderNode).
func renderLossless(n *Node, pretty bool) string {
	var sb strings.Builder
	renderLosslessInto(&sb, n, "", pretty)
	return sb.String()
}

func renderLosslessInto(sb *strings.Builder, n *Node, indent string, pretty bool) {
	colon := ":"
	if pretty {
		colon = ": "
	}
	switch n.Type {
	case NNull:
		sb.WriteString("null")
	case NBool:
		if n.Bool {
			sb.WriteString("true")
		} else {
			sb.WriteString("false")
		}
	case NString:
		sb.WriteString(quoteString(n.Str))
	case NNumber:
		if n.Num != nil {
			sb.WriteString(n.Num.Lexeme)
		} else {
			b, err := json.Marshal(n.f64)
			if err != nil {
				sb.WriteString("null")
			} else {
				sb.Write(b)
			}
		}
	case NObject:
		if len(n.Members) == 0 {
			sb.WriteString("{}")
			return
		}
		childIndent := indent + "  "
		sb.WriteByte('{')
		for i, mem := range n.Members {
			if i > 0 {
				sb.WriteByte(',')
			}
			if pretty {
				sb.WriteByte('\n')
				sb.WriteString(childIndent)
			}
			sb.WriteString(quoteString(mem.Key.Str))
			sb.WriteString(colon)
			renderLosslessInto(sb, mem.Value, childIndent, pretty)
		}
		if pretty {
			sb.WriteByte('\n')
			sb.WriteString(indent)
		}
		sb.WriteByte('}')
	case NArray:
		if len(n.Elems) == 0 {
			sb.WriteString("[]")
			return
		}
		childIndent := indent + "  "
		sb.WriteByte('[')
		for i, el := range n.Elems {
			if i > 0 {
				sb.WriteByte(',')
			}
			if pretty {
				sb.WriteByte('\n')
				sb.WriteString(childIndent)
			}
			renderLosslessInto(sb, el, childIndent, pretty)
		}
		if pretty {
			sb.WriteByte('\n')
			sb.WriteString(indent)
		}
		sb.WriteByte(']')
	}
}

// firstDuplicate walks a subtree and returns the first repeated object key
// and the span of its token, or nil when every object has unique keys.
func firstDuplicate(n *Node) (string, Span, bool) {
	switch n.Type {
	case NObject:
		seen := make(map[string]bool, len(n.Members))
		for _, mem := range n.Members {
			if seen[mem.Key.Str] {
				return mem.Key.Str, mem.Key.Span, true
			}
			seen[mem.Key.Str] = true
			if k, sp, dup := firstDuplicate(mem.Value); dup {
				return k, sp, true
			}
		}
	case NArray:
		for _, el := range n.Elems {
			if k, sp, dup := firstDuplicate(el); dup {
				return k, sp, true
			}
		}
	}
	return "", Span{}, false
}

// RenderNode serializes a node using the selected output strategy.
func RenderNode(n *Node, strategy OutputStrategy, pretty bool) (string, error) {
	if n == nil {
		return "", errors.New("failure json encode: nil node")
	}
	switch strategy {
	case StrategyCanonical:
		return marshalJSON(canonicalValue(n), pretty)
	case StrategyStandard:
		if key, sp, dup := firstDuplicate(n); dup {
			return "", &DuplicateKeyError{Key: key, Span: sp}
		}
		return marshalJSON(canonicalValue(n), pretty)
	default: // lossless
		// Compact output of a fully referenced subtree is the verbatim
		// source text: an exact, byte-identical round trip. For the
		// document root the entire source (including surrounding
		// whitespace) is returned.
		if !pretty && n.doc != nil {
			if n.root {
				return n.doc.src, nil
			}
			return n.doc.src[n.Span.ByteStart:n.Span.ByteEnd], nil
		}
		return renderLossless(n, pretty), nil
	}
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

// DupInfo describes one object whose key set contains duplicates.
type DupInfo struct {
	Key   string
	Span  Span
	Count int
	Extra int
}

// Diagnostics summarizes precision and duplicate-key findings for a result.
type Diagnostics struct {
	Strategy  OutputStrategy
	DupGroups []DupInfo
	DupExtra  int // total extra members across all flagged objects
	// BigNumbers counts referenced number literals that cannot be exactly
	// represented as float64.
	BigNumbers int
	// BeyondSafeInteger counts referenced integer literals with |n| > 2^53.
	BeyondSafeInteger int
	// totalNodes counts nodes walked (referenced + synthetic).
	totalNodes int
}

// HasDuplicates reports whether any duplicate key was observed.
func (d Diagnostics) HasDuplicates() bool { return d.DupExtra > 0 }

// RejectError builds the standard-strategy error for the first duplicate.
func (d Diagnostics) RejectError() error {
	if len(d.DupGroups) == 0 {
		return nil
	}
	g := d.DupGroups[0]
	return &DuplicateKeyError{Key: g.Key, Span: g.Span}
}

// CollectDiagnostics walks the result node and records precision / duplicate
// key findings according to the active output strategy.
func CollectDiagnostics(n *Node, strategy OutputStrategy) Diagnostics {
	d := Diagnostics{Strategy: strategy}
	var walk func(*Node)
	walk = func(x *Node) {
		if x == nil {
			return
		}
		d.totalNodes++
		switch x.Type {
		case NNumber:
			if x.Num != nil {
				nv := x.Num
				if nv.LosesFloat64Precision() {
					d.BigNumbers++
				}
				if nv.BeyondSafeInteger() {
					d.BeyondSafeInteger++
				}
			}
		case NObject:
			counts := map[string]int{}
			for _, mem := range x.Members {
				counts[mem.Key.Str]++
			}
			// Record groups in document order of the second occurrence.
			seen := map[string]bool{}
			for _, mem := range x.Members {
				k := mem.Key.Str
				if counts[k] > 1 && !seen[k] {
					seen[k] = true
					d.DupGroups = append(d.DupGroups, DupInfo{
						Key: k, Span: mem.Key.Span,
						Count: counts[k], Extra: counts[k] - 1,
					})
					d.DupExtra += counts[k] - 1
				}
			}
			for _, mem := range x.Members {
				walk(mem.Value)
			}
		case NArray:
			for _, el := range x.Elems {
				walk(el)
			}
		}
	}
	walk(n)
	return d
}

// status pieces are composed by Engine via StatusLine.

// StatusLine renders a one-line, human-readable diagnostic summary.
// alert is true when the line reports a hard error (standard rejection).
func (d Diagnostics) StatusLine() (text string, alert bool) {
	name := string(d.Strategy)
	if name == "" {
		name = string(StrategyLossless)
	}
	var b strings.Builder
	b.WriteString(name)

	switch d.Strategy {
	case StrategyStandard:
		if d.DupExtra > 0 && len(d.DupGroups) > 0 {
			g := d.DupGroups[0]
			fmt.Fprintf(&b, " · REJECTED duplicate key %q at %d:%d (byte %d)",
				g.Key, g.Span.Line, g.Span.Col, g.Span.ByteStart)
			return b.String(), true
		}
		b.WriteString(" · keys unique")
		if d.BigNumbers > 0 {
			fmt.Fprintf(&b, " · PRECISION LOSS: %d non-float64 number(s) (%d >2^53)",
				d.BigNumbers, d.BeyondSafeInteger)
			return b.String(), true
		}
	case StrategyCanonical:
		if d.DupExtra > 0 {
			fmt.Fprintf(&b, " · %d duplicate key occurrence(s) collapsed (last wins)", d.DupExtra)
		}
		if d.BigNumbers > 0 {
			fmt.Fprintf(&b, " · PRECISION LOSS: %d number(s) not exactly float64 (%d integers >2^53)",
				d.BigNumbers, d.BeyondSafeInteger)
		}
	default: // lossless
		if d.DupExtra > 0 {
			fmt.Fprintf(&b, " · %d duplicate key occurrence(s) kept in order", d.DupExtra)
		}
		if d.BigNumbers > 0 {
			fmt.Fprintf(&b, " · %d high-precision number(s) preserved (%d integers >2^53)",
				d.BigNumbers, d.BeyondSafeInteger)
		}
	}
	if d.DupExtra == 0 && d.BigNumbers == 0 {
		b.WriteString(" · no precision/duplicate-key issues")
	}
	return b.String(), false
}

// ---------------------------------------------------------------------------
// Tree navigation (referenced query results)
// ---------------------------------------------------------------------------

// nodeField returns the value of key name. Like encoding/json maps, the
// last member wins when duplicate keys exist; callers needing duplicates
// walk Members directly.
func nodeField(n *Node, name string) (*Node, bool) {
	if n == nil || n.Type != NObject {
		return syntheticNull(), false
	}
	for i := len(n.Members) - 1; i >= 0; i-- {
		if n.Members[i].Key.Str == name {
			return n.Members[i].Value, true
		}
	}
	return syntheticNull(), false
}

// nodeIndex returns array element idx, mirroring legacy getItem's
// out-of-range behaviour with a synthetic null node.
func nodeIndex(n *Node, idx int) *Node {
	if n == nil || n.Type != NArray || idx < 0 || idx >= len(n.Elems) {
		return syntheticNull()
	}
	return n.Elems[idx]
}

// nodeStep applies one legacy keyword (`name` or "[N]") to a node.
// An empty keyword leaves the node in place, mirroring getItem("").
func nodeStep(n *Node, keyword string) *Node {
	if keyword == "" {
		return n
	}
	if m := reArrayIndex.FindStringSubmatch(keyword); len(m) > 0 {
		idx, _ := strconv.Atoi(m[1])
		return nodeIndex(n, idx)
	}
	v, _ := nodeField(n, keyword)
	return v
}
