package jid

import (
	"bytes"
	"math/big"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

// ---------------------------------------------------------------------------
// Parser & spans
// ---------------------------------------------------------------------------

func TestParseDocumentSpans(t *testing.T) {
	assert := assert.New(t)

	// Multibyte key exercises byte vs rune offsets together.
	src := `{"名前":1,"b":[true,null,"x"]}`
	root, err := ParseDocument(src)
	assert.Nil(err)
	assert.Equal(NObject, root.Type)
	assert.Equal(0, root.Span.ByteStart)
	assert.Equal(len(src), root.Span.ByteEnd)
	assert.Equal(0, root.Span.RuneStart)
	assert.Equal(len([]rune(src)), root.Span.RuneEnd)

	// First key: `"名前"` starts at byte 1; each Japanese rune is 3 bytes.
	key := root.Members[0].Key
	assert.Equal(`名前`, key.Str)
	assert.Equal(1, key.Span.ByteStart)
	assert.Equal(9, key.Span.ByteEnd) // '"' + 2*3 bytes + '"'
	assert.Equal(1, key.Span.RuneStart)
	assert.Equal(5, key.Span.RuneEnd)
	txt, ok := key.SourceText()
	assert.True(ok)
	assert.Equal(`"名前"`, txt)

	// Value "1" is at byte 10 (closing quote at 8, ':' at 9),
	// rune index 6, column 7.
	num := root.Members[0].Value
	assert.Equal(NNumber, num.Type)
	assert.Equal(10, num.Span.ByteStart)
	assert.Equal(6, num.Span.RuneStart)
	assert.Equal(1, num.Span.Line)
	assert.Equal(7, num.Span.Col)

	// Array elements keep their own spans.
	arr := root.Members[1].Value
	assert.Equal(NArray, arr.Type)
	assert.Len(arr.Elems, 3)
	assert.Equal(NBool, arr.Elems[0].Type)
	assert.Equal(true, arr.Elems[0].Bool)
	assert.Equal(NNull, arr.Elems[1].Type)
	txt, ok = arr.Elems[2].SourceText()
	assert.True(ok)
	assert.Equal(`"x"`, txt)
}

func TestParseDocumentLineCol(t *testing.T) {
	assert := assert.New(t)

	src := "{\n  \"a\": {\n    \"b\": 7\n  }\n}"
	root, err := ParseDocument(src)
	assert.Nil(err)
	inner := root.Members[0].Value.Members[0].Value
	bi, _ := inner.Num.BigInt()
	assert.Equal(int64(7), bi.Int64())
	assert.Equal(3, inner.Span.Line)
	assert.Equal(10, inner.Span.Col) // `    "b": ` then 7 at column 10
}

func TestParseDocumentErrors(t *testing.T) {
	assert := assert.New(t)

	for _, bad := range []string{
		``,
		`{"a":}`,
		`{"a":1}x`,
		`01`,
		`1.`,
		`1.e2`,
		`{"a" 1}`,
		`[1,]`,
		`{"a":"b"`,
		`tru`,
	} {
		_, err := ParseDocument(bad)
		assert.Error(err, "expected parse error for %q", bad)
	}
}

func TestParseObjectPreservesOrderAndDuplicates(t *testing.T) {
	assert := assert.New(t)

	root, err := ParseDocument(`{"z":1,"a":2,"z":3,"m":4}`)
	assert.Nil(err)
	var keys []string
	for _, mem := range root.Members {
		keys = append(keys, mem.Key.Str)
	}
	assert.Equal([]string{"z", "a", "z", "m"}, keys)

	// Duplicate key spans point at the exact tokens.
	second := root.Members[2].Key
	assert.Equal(`"z"`, mustSource(t, second))
	assert.Equal(13, second.Span.ByteStart)
	assert.Equal(1, second.Span.Line)
	assert.Equal(14, second.Span.Col)
}

// ---------------------------------------------------------------------------
// NumberValue: lexeme + big.Int / big.Rat / decimal evaluation
// ---------------------------------------------------------------------------

func TestNumberValueEvaluations(t *testing.T) {
	assert := assert.New(t)

	root, err := ParseDocument(`{"i":9007199254740993,"r":0.10000000000000001,"half":0.5,"exp":1E3,"scale":1.20}`)
	assert.Nil(err)
	get := func(k string) *NumberValue {
		v, _ := nodeField(root, k)
		return v.Num
	}

	// big.Int exact
	i := get("i")
	bi, isInt := i.BigInt()
	assert.True(isInt)
	assert.Equal("9007199254740993", bi.String())
	assert.True(i.BeyondSafeInteger())
	assert.True(i.LosesFloat64Precision())

	// big.Rat exact decimal
	r := get("r")
	br, ok := r.BigRat()
	assert.True(ok)
	want, _ := new(big.Rat).SetString("10000000000000001/100000000000000000")
	assert.Equal(0, br.Cmp(want))
	assert.True(r.LosesFloat64Precision())

	// decimal (big.Float) keeps the value at 17 fractional digits
	df, ok := r.Decimal()
	assert.True(ok)
	assert.Equal("0.10000000000000001", df.Text('f', 17))

	// dyadic fractions are float64-exact
	half := get("half")
	_, ok = half.BigRat()
	assert.True(ok)
	assert.False(half.LosesFloat64Precision())
	f, ok := half.Float64()
	assert.True(ok)
	assert.Equal(0.5, f)

	// exponent + scale
	exp := get("exp")
	assert.True(exp.Valid())
	eint, isInt := exp.BigInt()
	assert.True(isInt)
	assert.Equal("1000", eint.String())
	sc := get("scale")
	assert.Equal(2, sc.Scale())
	assert.False(sc.IsInteger())
}

func TestNumberValueIntegersAround2Pow53(t *testing.T) {
	assert := assert.New(t)

	root, err := ParseDocument(`[9007199254740992,9007199254740993,-9007199254740993]`)
	assert.Nil(err)
	// 2^53 is not a "safe integer" but it is exactly representable as
	// float64, so it triggers the boundary flag without precision loss.
	assert.True(root.Elems[0].Num.BeyondSafeInteger())
	assert.False(root.Elems[0].Num.LosesFloat64Precision())
	assert.True(root.Elems[1].Num.BeyondSafeInteger())
	assert.True(root.Elems[1].Num.LosesFloat64Precision())
	assert.True(root.Elems[2].Num.BeyondSafeInteger())
	assert.True(root.Elems[2].Num.LosesFloat64Precision())

	bi, _ := root.Elems[2].Num.BigInt()
	assert.Equal("-9007199254740993", bi.String())
}

// ---------------------------------------------------------------------------
// Round trip guarantees
// ---------------------------------------------------------------------------

func TestLosslessRoundTrip(t *testing.T) {
	assert := assert.New(t)

	// Compact lossless output of the root is the verbatim source text —
	// whitespace, number lexemes and duplicate keys survive byte-for-byte.
	for _, src := range []string{
		`{"big":9007199254740993,"small":1.0000000000000001,"dup":1,"dup":2}`,
		`{ "big" : 9007199254740993 , "x": 0.10000000000000001 }` + "\n  ",
		`[1E100, -0.00000000000000001, 1.0, 100]`,
	} {
		jm := managerFrom(t, src)
		out, _, _, err := jm.Get(NewQueryWithString("."), true)
		assert.Nil(err)
		assert.Equal(src, out, "lossless compact output must be verbatim source")
	}
}

func TestLosslessPrettyKeepsLexemesAndDuplicates(t *testing.T) {
	assert := assert.New(t)

	jm := managerFrom(t, `{"x":1.000,"y":1E3,"dup":1,"dup":2}`)
	out, _, _, err := jm.GetPretty(NewQueryWithString("."), true)
	assert.Nil(err)
	assert.Equal(`{
  "x": 1.000,
  "y": 1E3,
  "dup": 1,
  "dup": 2
}`, out)
}

func TestCanonicalStrategyCollapsesAndConverts(t *testing.T) {
	assert := assert.New(t)

	jm := managerFrom(t, `{"big":9007199254740993,"d":0.10000000000000001,"dup":1,"dup":2}`)
	jm.SetOutputStrategy(StrategyCanonical)

	out, _, _, err := jm.Get(NewQueryWithString("."), true)
	assert.Nil(err)
	// float64 conversion changes the integer, prints the decimal as 0.1,
	// duplicate keys collapse to the last value, keys sorted.
	assert.Equal(`{"big":9007199254740992,"d":0.1,"dup":2}`, out)
}

func TestStandardStrategyRejectsDuplicates(t *testing.T) {
	assert := assert.New(t)

	jm := managerFrom(t, `{"ok":1,"dup":{"a":1,"a":2}}`)
	jm.SetOutputStrategy(StrategyStandard)

	// Root subtree (and a unique subtree) render canonically.
	out, _, _, err := jm.Get(NewQueryWithString(".ok"), true)
	assert.Nil(err)
	assert.Equal(`1`, out)

	// Selecting the duplicate-bearing object (or root) fails with the
	// duplicate key error carrying source location.
	_, _, _, err = jm.Get(NewQueryWithString(".dup"), true)
	dkErr, isDup := err.(*DuplicateKeyError)
	if assert.True(isDup) {
		assert.Equal("a", dkErr.Key)
		assert.Equal(1, dkErr.Span.Line)
		// `{"ok":1,"dup":{"a":1,` is 21 bytes, then the second `"a"`.
		assert.Equal(21, dkErr.Span.ByteStart)
		assert.Equal(22, dkErr.Span.Col)
	}

	_, _, _, err = jm.GetPretty(NewQueryWithString("."), true)
	assert.Error(err)
}

func TestStrategySwitchIsLive(t *testing.T) {
	assert := assert.New(t)

	jm := managerFrom(t, `{"dup":1,"dup":2}`)
	q := NewQueryWithString(".")

	lossless, _, _, _ := jm.Get(q, true)
	assert.Equal(`{"dup":1,"dup":2}`, lossless)

	jm.SetOutputStrategy(StrategyCanonical)
	canon, _, _, _ := jm.Get(q, true)
	assert.Equal(`{"dup":2}`, canon)

	jm.SetOutputStrategy(StrategyStandard)
	_, _, _, err := jm.Get(q, true)
	assert.Error(err)

	jm.SetOutputStrategy(StrategyLossless)
	again, _, _, _ := jm.Get(q, true)
	assert.Equal(lossless, again)
}

func TestNewEngineRejectsUnknownStrategy(t *testing.T) {
	assert := assert.New(t)

	_, err := NewEngine(bytes.NewBufferString(`{}`), &EngineAttribute{OutputMode: "bogus"})
	assert.Error(err)
}

func TestParseOutputStrategyNames(t *testing.T) {
	assert := assert.New(t)

	for name, want := range map[string]OutputStrategy{
		"":          StrategyLossless,
		"lossless":  StrategyLossless,
		"CANONICAL": StrategyCanonical,
		"standard":  StrategyStandard,
		"strict":    StrategyStandard,
	} {
		got, err := ParseOutputStrategy(name)
		assert.Nil(err)
		assert.Equal(want, got)
	}
	_, err := ParseOutputStrategy("nope")
	assert.Error(err)
}

// ---------------------------------------------------------------------------
// Query results reference the original tree (byte/rune traceability)
// ---------------------------------------------------------------------------

func TestLegacyQueryResultReferencesSourceNode(t *testing.T) {
	assert := assert.New(t)

	src := `{"名前":[9007199254740993,0.10000000000000001]}`
	jm := managerFrom(t, src)

	// Confirmed legacy navigation to the big integer.
	q := NewQueryWithString(".名前[0]")
	jm.GetFilteredData(q, true)
	n := jm.LastResultNode()
	if assert.NotNil(n) {
		assert.False(n.Synthetic)
		txt, ok := n.SourceText()
		assert.True(ok)
		assert.Equal("9007199254740993", txt)
		// {"名前":[ -> '9' is byte 11, rune 7 (two 3-byte runes).
		assert.Equal(11, n.Span.ByteStart)
		assert.Equal(7, n.Span.RuneStart)
	}

	// Lossless output of that node is its original lexeme.
	out, _, _, err := jm.Get(q, true)
	assert.Nil(err)
	assert.Equal("9007199254740993", out)

	// Canonical output changes it (diagnosable), same node reference.
	jm.SetOutputStrategy(StrategyCanonical)
	out, _, _, err = jm.Get(q, true)
	assert.Nil(err)
	assert.Equal("9007199254740992", out)
}

func TestLegacyDuplicateKeyNavigationLastWins(t *testing.T) {
	assert := assert.New(t)

	// Field access keeps go-jmespath/encoding/json compatibility: last
	// member wins, but its node still references the source span.
	jm := managerFrom(t, `{"a":[1,10],"a":{"b":2}}`)
	q := NewQueryWithString(".a.b")
	jm.GetFilteredData(q, true)
	n := jm.LastResultNode()
	assert.False(n.Synthetic)
	assert.Equal("2", mustSource(t, n))

	out, _, _, err := jm.Get(q, true)
	assert.Nil(err)
	assert.Equal(`2`, out)
}

func TestWildcardQueryElementsReferenceSource(t *testing.T) {
	assert := assert.New(t)

	jm := managerFrom(t, `{"items":[{"id":9007199254740993},{"id":7}]}`)
	q := NewQueryWithString(".items[*].id")
	jm.GetFilteredData(q, true)
	n := jm.LastResultNode()
	if assert.NotNil(n) {
		// The projected array is synthetic, but each element is a direct
		// reference to an original-tree number node.
		assert.True(n.Synthetic)
		assert.Len(n.Elems, 2)
		assert.False(n.Elems[0].Synthetic)
		assert.Equal("9007199254740993", mustSource(t, n.Elems[0]))
		assert.False(n.Elems[1].Synthetic)
		assert.Equal("7", mustSource(t, n.Elems[1]))
	}
}

func TestJMESPathFunctionResultIsSynthetic(t *testing.T) {
	assert := assert.New(t)

	jm := managerFrom(t, `{"users":[{"name":"a"},{"name":"b"}],"count":2}`)
	q := NewQueryWithString(". | keys(@)")
	jm.GetFilteredData(q, true)
	n := jm.LastResultNode()
	if assert.NotNil(n) {
		assert.True(n.Synthetic, "function results have no source span")
	}

	// Regular JMESPath behaviour still works through go-jmespath.
	r, _, _, err := jm.Get(NewQueryWithString(".users | length(@)"), true)
	assert.Nil(err)
	assert.Equal("2", r)
}

// ---------------------------------------------------------------------------
// Diagnostics / status line
// ---------------------------------------------------------------------------

func TestDiagnostics(t *testing.T) {
	assert := assert.New(t)

	jm := managerFrom(t, `{"big":9007199254740993,"fine":1,"dup":1,"dup":2,"nested":{"x":0.10000000000000001}}`)

	jm.GetFilteredData(NewQueryWithString("."), true)
	d := jm.LastDiagnostics()
	assert.Equal(StrategyLossless, d.Strategy)
	assert.Equal(1, d.DupExtra)
	if assert.Len(d.DupGroups, 1) {
		assert.Equal("dup", d.DupGroups[0].Key)
	}
	assert.Equal(2, d.BigNumbers) // big integer + high-precision decimal
	assert.Equal(1, d.BeyondSafeInteger)

	line, alert := d.StatusLine()
	assert.False(alert)
	assert.Contains(line, "lossless")
	assert.Contains(line, "duplicate key occurrence(s) kept")
	assert.Contains(line, "high-precision number(s) preserved")

	jm.SetOutputStrategy(StrategyCanonical)
	jm.GetFilteredData(NewQueryWithString("."), true)
	line, alert = jm.LastDiagnostics().StatusLine()
	assert.False(alert)
	assert.Contains(line, "collapsed (last wins)")
	assert.Contains(line, "PRECISION LOSS")

	jm.SetOutputStrategy(StrategyStandard)
	jm.GetFilteredData(NewQueryWithString("."), true)
	line, alert = jm.LastDiagnostics().StatusLine()
	assert.True(alert)
	assert.Contains(line, "REJECTED duplicate key")
}

func TestDiagnosticsClean(t *testing.T) {
	assert := assert.New(t)

	jm := managerFrom(t, `{"a":1,"b":[0.5,100]}`)
	jm.GetFilteredData(NewQueryWithString("."), true)
	line, alert := jm.LastDiagnostics().StatusLine()
	assert.False(alert)
	assert.Contains(line, "no precision/duplicate-key issues")
}

func TestDuplicateRejectionNoticeContents(t *testing.T) {
	assert := assert.New(t)

	e := getEngine(`{"a":1,"a":2}`, ".")
	e.manager.SetOutputStrategy(StrategyStandard)
	rows := e.getContents()
	joined := strings.Join(rows, "\n")
	assert.Contains(joined, `rejects duplicate keys`)
	assert.Contains(joined, `"a" appears 2 times`)
	assert.True(e.statusAlert)
	assert.Contains(e.statusLine, "REJECTED")
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func managerFrom(t *testing.T, src string) *JsonManager {
	t.Helper()
	jm, err := NewJsonManager(bytes.NewBufferString(src))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	return jm
}

func mustSource(t *testing.T, n *Node) string {
	t.Helper()
	s, ok := n.SourceText()
	if !ok {
		t.Fatalf("node has no source text: %v", n)
	}
	return s
}
