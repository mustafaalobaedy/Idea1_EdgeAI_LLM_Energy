"""Paper equations as native Word math (OMML), so they open in Word's equation editor.

Draft syntax: {{EQ:name}} on its own line inserts EQS[name] as a numbered equation; [eq:name] in text becomes (n).
"""

M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'


def r(t, plain=False):
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ''
    t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return f'<m:r>{sty}<m:t xml:space="preserve">{t}</m:t></m:r>'


def sub(b, s):
    return f'<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>'


def sup(b, s):
    return f'<m:sSup><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>'


def subsup(b, s, p):
    return f'<m:sSubSup><m:e>{b}</m:e><m:sub>{s}</m:sub><m:sup>{p}</m:sup></m:sSubSup>'


def frac(n, d):
    return f'<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>'


def nary(ch, lo, hi, e):
    hide = '' if hi else '<m:supHide m:val="1"/>'
    return (f'<m:nary><m:naryPr><m:chr m:val="{ch}"/><m:limLoc m:val="undOvr"/>{hide}</m:naryPr>'
            f'<m:sub>{lo}</m:sub><m:sup>{hi}</m:sup><m:e>{e}</m:e></m:nary>')


def integral(lo, hi, e):
    return (f'<m:nary><m:naryPr><m:chr m:val="∫"/><m:limLoc m:val="subSup"/></m:naryPr>'
            f'<m:sub>{lo}</m:sub><m:sup>{hi}</m:sup><m:e>{e}</m:e></m:nary>')


def paren(e, beg='(', end=')'):
    return f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr><m:e>{e}</m:e></m:d>'


def bar(e):
    return f'<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>{e}</m:e></m:bar>'


def omath(*parts):
    return f'<m:oMath xmlns:m="{M}">{"".join(parts)}</m:oMath>'


t0, t1 = sub(r('t'), r('0', plain=True)), sub(r('t'), r('1', plain=True))
Pm = sub(r('P'), r('module', plain=True))
Pt = Pm + paren(r('t'))
one = r('𝟙', plain=True)
ind = one + paren(sub(r('ŷ'), r('q')) + r(' = ') + sub(r('y'), r('q')), '[', ']')
sumQ = lambda e: nary('∑', r('q ∈ Q'), '', e)

EQS = {
    # P_module(t) = Σ_r V_r(t) I_r(t)
    'pmodule': omath(Pt, r(' = '), nary('∑', r('r'), '', sub(r('V'), r('r')) + paren(r('t')) + r(' ') + sub(r('I'), r('r')) + paren(r('t')))),
    # E_q = ∫_{t0}^{t1} P_module(t) dt
    'energy': omath(sub(r('E'), r('q')), r(' = '), integral(t0, t1, Pt + r(' d', plain=True) + r('t'))),
    # E_q^pre = ∫_{t0}^{t0+τ_q} P(t) dt,  E_q^dec = E_q − E_q^pre
    'split': omath(subsup(r('E'), r('q'), r('pre', plain=True)), r(' = '),
                   integral(t0, t0 + r(' + ') + sub(r('τ'), r('q')), Pt + r(' d', plain=True) + r('t')), r(',    '),
                   subsup(r('E'), r('q'), r('dec', plain=True)), r(' = '), sub(r('E'), r('q')), r(' − '),
                   subsup(r('E'), r('q'), r('pre', plain=True))),
    # E_q^dyn = E_q − P̄_idle (t1 − t0)
    'dyn': omath(subsup(r('E'), r('q'), r('dyn', plain=True)), r(' = '), sub(r('E'), r('q')), r(' − '),
                 sub(bar(r('P')), r('idle', plain=True)), r(' '), paren(t1 + r(' − ') + t0)),
    # A = (1/|Q|) Σ_q 1[ŷ_q = y_q]
    'acc': omath(r('A'), r(' = '), frac(r('1', plain=True), paren(r('Q'), '|', '|')), r(' '), sumQ(ind)),
    # JPCA = Σ_q E_q / Σ_q 1[ŷ_q = y_q]
    'jpca': omath(r('JPCA', plain=True), r(' = '), frac(sumQ(sub(r('E'), r('q'))), sumQ(ind))),
    # EDP = (1/|Q|) Σ_q E_q t_q
    'edp': omath(r('EDP', plain=True), r(' = '), frac(r('1', plain=True), paren(r('Q'), '|', '|')), r(' '),
                 sumQ(sub(r('E'), r('q')) + r(' ') + sub(r('t'), r('q')))),
    # E_casc = E_{s1} + E_{s2} + 1[a1 ≠ a2] E_L
    'cascade': omath(sub(r('E'), r('casc', plain=True)), r(' = '), sub(r('E'), sub(r('s'), r('1', plain=True))), r(' + '),
                     sub(r('E'), sub(r('s'), r('2', plain=True))), r(' + '),
                     one + paren(sub(r('a'), r('1', plain=True)) + r(' ≠ ') + sub(r('a'), r('2', plain=True)), '[', ']'),
                     r(' '), sub(r('E'), r('L'))),
}
