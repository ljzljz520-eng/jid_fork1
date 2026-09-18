package jid

// Tree-based evaluator for the navigation subset of JMESPath: identifiers,
// dot access, [N] indexes, [*] / .* projections and pipes between them.
// Results reuse parsed nodes (with their source spans); projections create
// synthetic array nodes whose elements still point at the original tree.
//
// Anything outside this subset (function calls, filters "[?...]",
// multi-select lists/hashes, quoted-identifier edge cases the tokenizer
// cannot split) is rejected so the caller falls back to go-jmespath and
// wraps the result in a fully synthetic tree.

import (
	"strconv"
	"strings"
)

type segKind int

const (
	segField segKind = iota
	segIndex
	segArrStar
	segObjStar
)

type pathSeg struct {
	kind segKind
	name string
	idx  int
}

// parsePathGroups splits expr into pipe-separated segment groups.
// Returns false when the expression uses syntax this evaluator does not
// support.
func parsePathGroups(expr string) ([][]pathSeg, bool) {
	s := strings.TrimSpace(expr)
	if s == "" {
		return nil, false
	}
	var groups [][]pathSeg
	var cur []pathSeg
	i := 0
	groupStarted := false // '@' or a first identifier already consumed
	for i < len(s) {
		// skip spaces between tokens
		for i < len(s) && s[i] == ' ' {
			i++
		}
		if i >= len(s) {
			break
		}
		c := s[i]
		switch {
		case c == '@':
			if groupStarted {
				return nil, false
			}
			i++
			groupStarted = true
		case c == '.':
			i++
			if i < len(s) && s[i] == '*' {
				cur = append(cur, pathSeg{kind: segObjStar})
				i++
				groupStarted = true
				continue
			}
			if i < len(s) && s[i] == '"' {
				name, next, ok := readQuotedIdent(s, i)
				if !ok {
					return nil, false
				}
				cur = append(cur, pathSeg{kind: segField, name: name})
				i = next
				groupStarted = true
				continue
			}
			name, next, ok := readBareIdent(s, i)
			if !ok {
				return nil, false
			}
			cur = append(cur, pathSeg{kind: segField, name: name})
			i = next
			groupStarted = true
		case c == '[':
			i++
			if i < len(s) && s[i] == '*' {
				i++
				if i >= len(s) || s[i] != ']' {
					return nil, false
				}
				i++
				cur = append(cur, pathSeg{kind: segArrStar})
				groupStarted = true
				continue
			}
			start := i
			for i < len(s) && s[i] >= '0' && s[i] <= '9' {
				i++
			}
			if i == start || i >= len(s) || s[i] != ']' {
				return nil, false // slices, filters, multi-select lists
			}
			idx, err := strconv.Atoi(s[start:i])
			if err != nil {
				return nil, false
			}
			i++ // consume ']'
			cur = append(cur, pathSeg{kind: segIndex, idx: idx})
			groupStarted = true
		case isIdentStart(c) && !groupStarted:
			// Bare identifier starting a group (e.g. "users[*].name").
			name, next, ok := readBareIdent(s, i)
			if !ok {
				return nil, false
			}
			cur = append(cur, pathSeg{kind: segField, name: name})
			i = next
			groupStarted = true
		case c == '|':
			if !groupStarted {
				return nil, false
			}
			groups = append(groups, cur)
			cur = nil
			i++
			groupStarted = false
		default:
			return nil, false
		}
	}
	if !groupStarted {
		return nil, false
	}
	groups = append(groups, cur)
	return groups, true
}

func isIdentStart(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isIdentPart(c byte) bool {
	return isIdentStart(c) || (c >= '0' && c <= '9')
}

// readBareIdent reads an identifier starting at s[i]. A leading dot (if
// present) must already have been consumed; here i points at the first name
// byte.
func readBareIdent(s string, i int) (string, int, bool) {
	start := i
	for i < len(s) && isIdentPart(s[i]) {
		i++
	}
	if start == i {
		return "", i, false
	}
	return s[start:i], i, true
}

// scanQuotedToken returns the end offset (exclusive) of a JSON-style quoted
// token beginning at s[start] == '"', honoring backslash escapes.
func scanQuotedToken(s string, start int) (int, bool) {
	i := start + 1
	for i < len(s) {
		switch s[i] {
		case '"':
			return i + 1, true
		case '\\':
			i += 2
			continue
		}
		i++
	}
	return 0, false
}

// readQuotedIdent reads a JMESPath quoted identifier "..." starting at the
// opening quote; returns the decoded name and the offset just past it.
func readQuotedIdent(s string, i int) (string, int, bool) {
	start := i
	end, ok := scanQuotedToken(s, i)
	if !ok {
		return "", 0, false
	}
	name, qerr := strconv.Unquote(s[start:end])
	if qerr != nil {
		return "", 0, false
	}
	return name, end, true
}

// evalPathOnTree evaluates a supported expression against the document tree.
func evalPathOnTree(expr string, root *Node) (*Node, bool) {
	groups, ok := parsePathGroups(expr)
	if !ok {
		return nil, false
	}
	cur := root
	for _, g := range groups {
		cur, ok = evalPathGroup(cur, g)
		if !ok {
			return nil, false
		}
	}
	return cur, true
}

// evalPathGroup applies one pipe group. It models JMESPath projection
// semantics closely enough for the navigation subset: a wildcard turns the
// intermediate into a projected list, subsequent field/index/wildcard
// segments operate per element and non-matching elements are dropped.
func evalPathGroup(root *Node, segs []pathSeg) (*Node, bool) {
	projected := false
	var items []*Node
	single := root

	for _, seg := range segs {
		if !projected {
			switch seg.kind {
			case segField:
				// Identifier on a non-object (including null) resolves to null.
				single, _ = nodeField(single, seg.name)
			case segIndex:
				single = nodeIndex(single, seg.idx)
			case segArrStar:
				if single != nil && single.Type == NArray {
					items = append(items, single.Elems...)
					projected = true
				} else {
					single = syntheticNull()
				}
			case segObjStar:
				if single != nil && single.Type == NObject {
					for _, mem := range single.Members {
						items = append(items, mem.Value)
					}
					projected = true
				} else {
					single = syntheticNull()
				}
			}
			continue
		}
		// Projection context: map over elements, dropping misses.
		next := make([]*Node, 0, len(items))
		switch seg.kind {
		case segField:
			for _, el := range items {
				if v, ok := nodeField(el, seg.name); ok {
					next = append(next, v)
				}
			}
		case segIndex:
			for _, el := range items {
				if el != nil && el.Type == NArray && seg.idx >= 0 && seg.idx < len(el.Elems) {
					next = append(next, el.Elems[seg.idx])
				}
			}
		case segArrStar:
			for _, el := range items {
				if el != nil && el.Type == NArray {
					next = append(next, el.Elems...)
				}
			}
		case segObjStar:
			for _, el := range items {
				if el != nil && el.Type == NObject {
					for _, mem := range el.Members {
						next = append(next, mem.Value)
					}
				}
			}
		}
		items = next
	}

	if projected {
		return newSyntheticArray(items), true
	}
	return single, true
}
