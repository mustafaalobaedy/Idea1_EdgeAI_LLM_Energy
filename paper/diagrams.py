"""Editable, black-and-white Word diagrams (DrawingML group shapes) for the paper.

Draft syntax: {{DIAGRAM:name|caption|label}} inserts DIAGRAMS[name]() as an inline drawing whose boxes and arrows
can be edited in Word (Transitional OOXML namespaces, matching the PeerJ template).
"""
from xml.sax.saxutils import escape

EMU_PT = 12700
A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
WPG_NS = 'http://schemas.microsoft.com/office/word/2010/wordprocessingGroup'
WPS_NS = 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
WP_NS = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
FONT = 'Times New Roman'


def esc(t):
    return escape(t, {'"': '&quot;'})


def _runs(text, size, bold=False):
    """**bold** and *italic* inline markup -> w:r elements."""
    import re
    out = []
    for j, part in enumerate(re.split(r'\*\*([^*]+)\*\*', text)):
        for i, x in enumerate(re.split(r'\*([^*]+)\*', part)):
            if not x:
                continue
            rpr = f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:cs="{FONT}"/>'
            if bold or j % 2 == 1:
                rpr += '<w:b/>'
            if i % 2 == 1:
                rpr += '<w:i/>'
            rpr += f'<w:sz w:val="{int(size * 2)}"/><w:szCs w:val="{int(size * 2)}"/>'
            out.append(f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(x)}</w:t></w:r>')
    return ''.join(out)


class Diagram:
    """Boxes with text and arrow lines, positioned in points. Black outlines, white or light-gray fills only."""

    def __init__(self, w_pt, h_pt, font_pt=9):
        self.w, self.h, self.font = w_pt, h_pt, font_pt
        self.items = []
        self.n = 0

    def _id(self):
        self.n += 1
        return self.n

    @staticmethod
    def _e(v):
        return int(round(v * EMU_PT))

    def box(self, x, y, w, h, lines, fill='FFFFFF', rounded=False, dashed=False, size=None):
        sid = self._id()
        size = size or self.font
        ps = ''.join(f'<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="{int((size + 2) * 20)}" w:lineRule="exact"/>'
                     f'<w:jc w:val="center"/></w:pPr>{_runs(ln, size)}</w:p>' for ln in lines)
        dash = '<a:prstDash val="dash"/>' if dashed else ''
        geom = 'roundRect' if rounded else 'rect'
        av = '<a:avLst><a:gd name="adj" fmla="val 12000"/></a:avLst>' if rounded else '<a:avLst/>'
        self.items.append(
            f'<wps:wsp><wps:cNvPr id="{sid}" name="Box {sid}"/><wps:cNvSpPr/><wps:spPr>'
            f'<a:xfrm><a:off x="{self._e(x)}" y="{self._e(y)}"/><a:ext cx="{self._e(w)}" cy="{self._e(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geom}">{av}</a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
            f'<a:ln w="9525"><a:solidFill><a:srgbClr val="000000"/></a:solidFill>{dash}</a:ln></wps:spPr>'
            f'<wps:txbx><w:txbxContent>{ps}</w:txbxContent></wps:txbx>'
            f'<wps:bodyPr rot="0" vert="horz" wrap="square" lIns="18000" tIns="18000" rIns="18000" bIns="18000" '
            f'anchor="ctr" anchorCtr="0" upright="1"><a:noAutofit/></wps:bodyPr></wps:wsp>')

    def text(self, x, y, w, h, s, size=None, align='left'):
        sid = self._id()
        size = size or self.font
        self.items.append(
            f'<wps:wsp><wps:cNvPr id="{sid}" name="Label {sid}"/><wps:cNvSpPr txBox="1"/><wps:spPr>'
            f'<a:xfrm><a:off x="{self._e(x)}" y="{self._e(y)}"/><a:ext cx="{self._e(w)}" cy="{self._e(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></wps:spPr>'
            f'<wps:txbx><w:txbxContent><w:p><w:pPr><w:spacing w:before="0" w:after="0"/><w:jc w:val="{align}"/></w:pPr>'
            f'{_runs(s, size)}</w:p></w:txbxContent></wps:txbx>'
            f'<wps:bodyPr rot="0" vert="horz" wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t" anchorCtr="0" '
            f'upright="1"><a:noAutofit/></wps:bodyPr></wps:wsp>')

    def arrow(self, x1, y1, x2, y2, head_end=True, head_start=False, dashed=False):
        sid = self._id()
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        flip = (' flipH="1"' if x2 < x1 else '') + (' flipV="1"' if y2 < y1 else '')
        dash = '<a:prstDash val="dash"/>' if dashed else ''
        he = '<a:tailEnd type="triangle" w="med" len="med"/>' if head_end else ''
        hs = '<a:headEnd type="triangle" w="med" len="med"/>' if head_start else ''
        self.items.append(
            f'<wps:wsp><wps:cNvPr id="{sid}" name="Arrow {sid}"/><wps:cNvCnPr/><wps:spPr>'
            f'<a:xfrm{flip}><a:off x="{self._e(x)}" y="{self._e(y)}"/><a:ext cx="{self._e(w)}" cy="{self._e(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="straightConnector1"><a:avLst/></a:prstGeom>'
            f'<a:ln w="9525"><a:solidFill><a:srgbClr val="000000"/></a:solidFill>{dash}{hs}{he}</a:ln></wps:spPr>'
            f'<wps:bodyPr/></wps:wsp>')

    def xml(self, pid, name):
        cx, cy = self._e(self.w), self._e(self.h)
        return (f'<w:r xmlns:w="{W_NS}"><w:drawing><wp:inline xmlns:wp="{WP_NS}" distT="0" distB="0" distL="0" distR="0">'
                f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
                f'<wp:docPr id="{pid}" name="{esc(name)}"/><wp:cNvGraphicFramePr/>'
                f'<a:graphic xmlns:a="{A_NS}"><a:graphicData uri="{WPG_NS}">'
                f'<wpg:wgp xmlns:wpg="{WPG_NS}" xmlns:wps="{WPS_NS}"><wpg:cNvGrpSpPr/><wpg:grpSpPr>'
                f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/><a:chOff x="0" y="0"/><a:chExt cx="{cx}" cy="{cy}"/></a:xfrm>'
                f'</wpg:grpSpPr>{"".join(self.items)}</wpg:wgp></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')


# ----------------------------------------------------------------------------- Fig. 1: measurement setup
def setup_diagram():
    """Harness -> dedicated Ollama server -> model on the GPU; power sampler reads the INA3221 rails; energy integration."""
    D = Diagram(460, 200, font_pt=9)
    G = 'EEEEEE'
    # left column: the benchmark harness
    D.box(0, 32, 105, 138, ['**Benchmark harness**', '(Python)', '', 'fixed item subsets', 'ARC, BoolQ, GSM8K', '',
                             'greedy decoding', 'one request at a time', '', 'idle-machine guard'], fill=G)
    # the Jetson module (outer box) with a two-line title
    D.box(140, 0, 320, 196, [''], fill='FFFFFF')
    D.text(145, 4, 310, 26, '**NVIDIA Jetson AGX Orin 32 GB module**  –  power mode set with nvpmodel: 15 W, 30 W, 40 W, or MAXN',
           size=8, align='center')
    D.box(160, 32, 130, 62, ['**Ollama 0.33 server**', '(dedicated instance)', 'llama.cpp, GGUF model', 'GPU + shared LPDDR5'])
    D.box(310, 32, 135, 62, ['**INA3221 sensors**', 'VDD_GPU_SOC, VDD_CPU_CV', 'VIN_SYS_5V0', 'VDDQ_VDD2_1V8AO'])
    D.box(160, 118, 130, 62, ['**Response**', 'text, token counts,', 'prefill and decode', 'durations'])
    D.box(310, 118, 135, 62, ['**Power sampler thread**', '50 Hz over hwmon', 'P(t) = Σ V_r(t) · I_r(t)', 'temperatures, GPU clock'])
    # arrows and their labels (labels sit clear of the arrow lines)
    D.arrow(105, 62, 160, 62)                      # request
    D.text(106, 46, 54, 12, 'request, t0', size=7, align='center')
    D.arrow(160, 150, 105, 150)                    # response back
    D.text(106, 153, 54, 12, 'response, t1', size=7, align='center')
    D.arrow(225, 94, 225, 118)                     # server -> response
    D.text(229, 100, 40, 12, 'tokens', size=7)
    D.arrow(377, 94, 377, 118)                     # sensors -> sampler
    D.text(381, 100, 50, 12, 'V, I per rail', size=7)
    D.arrow(310, 150, 290, 150, dashed=True)       # sampler trace joined with the response timestamps
    D.text(292, 136, 20, 12, 'P(t)', size=7)
    D.text(0, 182, 140, 14, 'E_q = ∫ P(t) dt over [t0, t1]', size=9)
    return D


DIAGRAMS = {'setup': setup_diagram}
