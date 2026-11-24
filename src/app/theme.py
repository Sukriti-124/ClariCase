"""Shared look for the intake app and the admin dashboard (Graphite & mint).

A dark graphite band runs across the top of the page; white cards start on
the band and overlap its lower edge. The band is painted as the app
background, so it needs no layout of its own. Cards are keyed containers
(st.container(key="card_…")), which Streamlit renders with the class
"st-key-card_…".
"""

import html as _html

BAND = "#1D2024"
BAND_MUTED = "#AEB3BA"
INK = "#17202A"
MUTED = "#5C6773"
LINE = "#DDE3E8"
LINE_SOFT = "#EDF1F4"
PAGE = "#F2F3F4"
ACCENT = "#0E8C7E"
ACCENT_SOFT = "#9ED3CB"
SOFT = "#E2F3F0"
MARK = "#D9F5E8"
OK = "#1A7F37"
OK_SOFT = "#E6F4EA"
WARN = "#B26B00"
WARN_SOFT = "#FFF4E0"
NEUTRAL = "#8C959F"

LOGO = ('<svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true">'
        f'<rect x="2" y="2" width="20" height="20" rx="5" fill="{ACCENT}"/>'
        '<path d="M15.5 8.2a5 5 0 1 0 0 7.6" fill="none" stroke="#fff" '
        'stroke-width="2.4" stroke-linecap="round"/>'
        '<circle cx="16.4" cy="12" r="1.6" fill="#8FF0DC"/></svg>')
ICON_LOCK = ('<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">'
             '<rect x="5" y="11" width="14" height="9" rx="2" fill="none" '
             'stroke="currentColor" stroke-width="1.8"/><path d="M8 11V8a4 4 0 '
             '0 1 8 0v3" fill="none" stroke="currentColor" stroke-width="1.8"/>'
             '</svg>')
ICON_CHECK = ('<svg width="13" height="13" viewBox="0 0 24 24" aria-hidden="true">'
              '<path d="M12 2l8 3v6c0 5-3.5 9-8 11-4.5-2-8-6-8-11V5z" '
              'fill="currentColor" opacity=".15"/><path d="M8.5 12l2.5 2.5 '
              '4.5-5" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round"/></svg>')

# Short codes for the team badge; names alone would collide (two "MT"s).
TEAM_CODES = {"T01": "CR", "T02": "DC", "T03": "CC", "T04": "BA", "T05": "TR",
              "T06": "MG", "T07": "SL", "T08": "VF", "T09": "PL", "T10": "PC",
              "T11": "DM"}


def css(band_px: int = 300) -> str:
    """Page styles. `band_px` is how far the dark band reaches down."""
    return f"""
<style>
  .stApp {{ background: linear-gradient(180deg, {BAND} 0, {BAND} {band_px}px,
                                        {PAGE} {band_px}px) no-repeat,
                        {PAGE}; }}
  /* Faint grid on the band, fading out towards its lower edge */
  .stApp {{ position: relative; }}
  .stApp::before {{ content: ""; position: absolute; top: 0; left: 0; right: 0;
      height: {band_px}px; pointer-events: none; z-index: 0;
      background-image:
        linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px);
      background-size: 36px 36px;
      -webkit-mask-image: linear-gradient(180deg, #000 0%, transparent 90%);
      mask-image: linear-gradient(180deg, #000 0%, transparent 90%); }}
  [data-testid="stAppViewContainer"] {{ position: relative; z-index: 1; }}
  .block-container {{ max-width: 1240px; padding-top: 0.6rem;
                      padding-bottom: 3rem; }}
  [data-testid="stStatusWidget"] {{ visibility: hidden; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stToolbar"] button {{ color: {BAND_MUTED}; }}

  /* Brand row */
  .cc-top {{ display: flex; align-items: center; gap: 10px; height: 60px;
             border-bottom: 1px solid rgba(255,255,255,.09); }}
  .cc-word {{ font-size: 17px; font-weight: 600; color: #fff; }}
  .cc-area {{ font-size: 13px; color: {BAND_MUTED}; margin-left: 6px;
              padding-left: 12px; border-left: 1px solid rgba(255,255,255,.15); }}
  .cc-env {{ margin-left: auto; font-size: 12.5px; color: {BAND_MUTED}; }}

  /* Tabs sit in the brand row, on the right */
  /* admin page links, sitting in the brand row */
  .st-key-cc_nav {{ margin-top: -64px; height: 44px; align-items: center;
                    position: relative; z-index: 2; }}
  .st-key-cc_nav [data-testid="stPageLink"] a {{ padding: 7px 12px;
      border-radius: 6px; background: transparent; }}
  .st-key-cc_nav [data-testid="stPageLink"] a span,
  .st-key-cc_nav [data-testid="stPageLink"] a p {{ color: {BAND_MUTED};
      font-size: 14px; }}
  .st-key-cc_nav [data-testid="stPageLink"] a:hover span,
  .st-key-cc_nav [data-testid="stPageLink"] a:hover p {{ color: #fff; }}
  .st-key-cc_nav [data-testid="stPageLink"] a[aria-current="page"] {{
      background: rgba(255,255,255,.10); }}
  .st-key-cc_nav [data-testid="stPageLink"] a[aria-current="page"] span,
  .st-key-cc_nav [data-testid="stPageLink"] a[aria-current="page"] p {{
      color: #fff; font-weight: 600; }}
  .stTabs {{ position: relative; }}
  .stTabs [role="tablist"] {{ position: absolute; top: -50px; right: 0;
                              gap: 4px; z-index: 5; border: 0 !important;
                              box-shadow: none !important; }}
  .stTabs .react-aria-SelectionIndicator {{ display: none; }}
  .stTabs [role="tablist"], .stTabs [role="tablist"] *,
  .stTabs [data-orientation="horizontal"] {{ border-bottom: 0 !important; }}
  .stTabs [role="tablist"]::before, .stTabs [role="tablist"]::after {{
      display: none !important; }}
  .stTabs [data-testid="stTab"] {{ padding: 7px 12px; border-radius: 6px;
                                   height: auto; border: 0; }}
  .stTabs [data-testid="stTab"] p {{ font-size: 14px; color: {BAND_MUTED}; }}
  .stTabs [data-testid="stTab"]:hover p {{ color: #fff; }}
  .stTabs [data-testid="stTab"][aria-selected="true"] {{
      background: rgba(255,255,255,.10); }}
  .stTabs [data-testid="stTab"][aria-selected="true"] p {{
      color: #fff; font-weight: 500; }}
  .stTabs [data-testid="stTabPanel"] {{ padding-top: 0; }}

  /* Inputs inside cards */
  [class*="st-key-card_"] textarea {{ background: #FCFCFD; font-size: 14.5px;
                                      line-height: 1.55; }}
  [class*="st-key-card_"] input {{ background: #FCFCFD; }}
  [class*="st-key-card_"] [data-testid="stFileUploaderDropzone"] {{
      background: #FCFCFD; border: 1px dashed {LINE}; }}
  [class*="st-key-card_"] [data-baseweb="textarea"],
  [class*="st-key-card_"] [data-baseweb="input"] {{ background: #FCFCFD;
      border: 1px solid {LINE}; }}

  /* Heading on the band */
  .cc-h1 {{ margin: 34px 0 0; font-size: 30px; font-weight: 600;
            letter-spacing: -0.01em; line-height: 1.2; color: #fff; }}
  .cc-sub {{ margin: 8px 0 26px; font-size: 15px; line-height: 1.5;
             color: {BAND_MUTED}; max-width: 640px; }}
  .cc-crumbs {{ display: flex; flex-wrap: wrap; list-style: none;
                margin: -10px 0 18px; padding: 0; font-size: 13px;
                color: {BAND_MUTED}; }}
  .cc-crumbs li {{ margin: 0; padding: 0; }}
  .cc-crumbs li + li::before {{ content: "/"; margin: 0 10px; opacity: .4; }}
  .cc-crumbs li.done {{ color: #D5D9DE; }}
  .cc-crumbs li.now {{ color: #fff; font-weight: 600; }}

  /* Cards */
  [class*="st-key-card_"] {{ background: #FFFFFF; border-radius: 10px;
      padding: 16px 20px 18px;
      box-shadow: 0 8px 28px rgba(20,30,40,.12), 0 1px 2px rgba(20,30,40,.06); }}
  .cc-ph {{ display: flex; justify-content: space-between; align-items: center;
            gap: 12px; font-size: 14px; font-weight: 600; color: {INK};
            padding-bottom: 12px; border-bottom: 1px solid {LINE_SOFT}; }}
  .cc-ph > span {{ font-weight: 400; font-size: 12.5px; color: {MUTED}; }}
  .cc-ph > span.cc-id {{ font-size: 14px; font-weight: 600; color: {INK}; }}
  .cc-label {{ font-size: 12px; font-weight: 600; color: {MUTED};
               margin: 16px 0 8px; }}
  .cc-muted {{ font-size: 13px; color: {MUTED}; }}
  .cc-empty {{ font-size: 14px; color: {MUTED}; padding: 22px 0 10px;
               line-height: 1.55; }}
  .cc-note {{ display: flex; gap: 8px; align-items: center; font-size: 12.5px;
              color: {MUTED}; }}
  .cc-note svg {{ color: {ACCENT}; flex: 0 0 auto; }}
  .cc-x {{ font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace;
           font-size: 12px; background: {LINE_SOFT}; padding: 0 5px;
           border-radius: 3px; color: {INK}; }}
  .cc-mono {{ font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace; }}

  /* Result */
  .cc-result-top {{ display: flex; justify-content: space-between;
                    align-items: center; gap: 16px; padding: 18px;
                    border-radius: 8px;
                    background: linear-gradient(180deg, {SOFT}, #FFFFFF); }}
  .cc-team {{ display: flex; gap: 14px; align-items: center; min-width: 0; }}
  .cc-avatar {{ width: 52px; height: 52px; border-radius: 9px; flex: 0 0 auto;
                display: grid; place-items: center; font-weight: 700;
                font-size: 16px; color: #fff; background: {ACCENT}; }}
  .cc-team-label {{ font-size: 12px; color: {MUTED}; }}
  .cc-team-name {{ font-size: 21px; font-weight: 600; line-height: 1.2;
                   color: {INK}; }}
  .cc-meta {{ display: flex; gap: 10px; align-items: center; margin-top: 7px;
              flex-wrap: wrap; }}
  .cc-id {{ font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace;
            font-size: 13px; font-weight: 500; color: {INK};
            user-select: all; cursor: text; }}
  [class*="st-key-card_"] [data-testid="stCode"] pre {{
      background: {SOFT}; border: 1px solid {ACCENT_SOFT}; border-radius: 8px; }}
  [class*="st-key-card_"] [data-testid="stCode"] code {{
      font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace;
      font-size: 15px; font-weight: 600; color: {INK}; }}
  .cc-track-note {{ margin: -4px 0 4px; font-size: 12.5px; }}
  .cc-status {{ display: inline-flex; align-items: center; gap: 6px;
                font-size: 12.5px; font-weight: 600; padding: 3px 9px;
                border-radius: 4px; color: {MUTED}; background: {LINE_SOFT};
                white-space: nowrap; }}
  .cc-status::before {{ content: ""; width: 7px; height: 7px;
                        border-radius: 50%; background: currentColor; }}
  .cc-s-ok {{ color: {OK}; background: {OK_SOFT}; }}
  .cc-s-warn {{ color: {WARN}; background: {WARN_SOFT}; }}
  .cc-gauge {{ position: relative; width: 84px; height: 84px; flex: 0 0 auto; }}
  .cc-gauge svg {{ width: 84px; height: 84px; transform: rotate(-90deg); }}
  .cc-gauge-v {{ position: absolute; inset: 0; display: grid;
                 place-items: center; text-align: center; }}
  .cc-gauge-v b {{ display: block; font-size: 19px; font-weight: 600;
                   line-height: 1; color: {INK}; }}
  .cc-gauge-v small {{ font-size: 10.5px; color: {MUTED}; }}
  .cc-also {{ font-size: 13px; color: {MUTED}; margin-top: 10px; }}

  /* Preview shown before anything is routed */
  .cc-ghost {{ opacity: .55; filter: grayscale(1); pointer-events: none;
               user-select: none; }}
  .cc-ghost .cc-avatar {{ background: #C9D1D8; }}
  .cc-ghost-line {{ height: 9px; border-radius: 5px; background: {LINE};
                    margin: 9px 0; }}
  .cc-ghost-cap {{ font-size: 13px; color: {MUTED}; text-align: center;
                   margin: 14px 0 2px; }}

  /* Bars */
  .cc-bar-row {{ display: grid; grid-template-columns: minmax(0,1fr) 110px 40px;
                 gap: 12px; align-items: center; margin: 7px 0;
                 font-size: 13px; color: {MUTED}; }}
  .cc-bar-row.top {{ color: {INK}; font-weight: 500; font-size: 14px; }}
  .cc-bar-name {{ overflow: hidden; text-overflow: ellipsis;
                  white-space: nowrap; }}
  .cc-track {{ height: 5px; background: {LINE_SOFT}; border-radius: 3px;
               overflow: hidden; }}
  .cc-fill {{ height: 100%; background: {ACCENT_SOFT}; border-radius: 3px; }}
  .cc-bar-row.top .cc-fill {{ background: {ACCENT}; }}
  .cc-bar-val {{ text-align: right; font-variant-numeric: tabular-nums; }}

  /* Why */
  .cc-why p {{ margin: 0 0 12px; font-size: 14px; line-height: 1.6;
               color: {INK}; }}
  .cc-excerpt {{ font-size: 13.5px; line-height: 1.75; color: #36414B;
                 background: #FCFCFD; border: 1px solid {LINE_SOFT};
                 border-left: 3px solid {ACCENT}; border-radius: 4px;
                 padding: 10px 12px; max-height: 220px; overflow-y: auto; }}
  .cc-excerpt mark {{ background: {MARK}; color: {INK}; padding: 0 2px;
                      border-radius: 2px; }}
  .cc-src {{ display: flex; align-items: center; gap: 6px; font-size: 12px;
             color: {MUTED}; margin-top: 12px; }}
  .cc-src svg {{ color: {OK}; }}

  /* Processing checklist */
  .cc-proc {{ list-style: none; margin: 12px 0 0; padding: 0; font-size: 14px; }}
  .cc-proc li {{ display: flex; gap: 10px; padding: 5px 0; color: {NEUTRAL}; }}
  .cc-proc li::before {{ content: ""; width: 16px; flex: 0 0 16px;
                         text-align: center; }}
  .cc-proc li.done {{ color: {INK}; }}
  .cc-proc li.done::before {{ content: "\\2713"; color: {ACCENT};
                              font-weight: 700; }}
  .cc-proc li.now {{ color: {INK}; font-weight: 600; }}
  .cc-proc li.now::before {{ content: "\\2022"; color: {ACCENT}; }}
  .cc-proc li.todo::before {{ content: "\\2013"; }}

  /* Stat strip */
  .cc-stats {{ display: grid; grid-template-columns: repeat(auto-fit,
               minmax(150px, 1fr)); margin: 4px 0; }}
  .cc-stat {{ padding: 4px 16px; border-left: 1px solid {LINE_SOFT}; }}
  .cc-stat:first-child {{ border-left: 0; padding-left: 0; }}
  .cc-stat-label {{ font-size: 12px; font-weight: 500; color: {MUTED}; }}
  .cc-stat-num {{ font-size: 24px; font-weight: 600; color: {INK};
                  margin-top: 2px; line-height: 1.2;
                  font-variant-numeric: tabular-nums; }}
  .cc-stat-foot {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}

  /* Track timeline */
  .cc-tl {{ list-style: none; margin: 4px 0 0; padding: 0; }}
  .cc-tl li {{ display: flex; gap: 12px; }}
  .cc-tl-rail {{ display: flex; flex-direction: column; align-items: center;
                 padding-top: 5px; }}
  .cc-tl-dot {{ width: 10px; height: 10px; border-radius: 50%;
                flex: 0 0 auto; }}
  .cc-tl-line {{ width: 2px; flex: 1 1 auto; min-height: 18px;
                 margin-top: 4px; border-radius: 1px; }}
  .cc-tl-body {{ padding-bottom: 14px; font-size: 14px; }}
  .cc-tl-title {{ font-weight: 500; color: {INK}; }}
  .cc-kv {{ display: grid; grid-template-columns: 120px 1fr; gap: 10px 14px;
            font-size: 14px; margin: 0; }}
  .cc-kv dt {{ color: {MUTED}; font-size: 14px; }}
  .cc-kv dd {{ margin: 0; color: {INK}; font-size: 14px; }}
  .cc-tag {{ display: inline-block; font-size: 13px; padding: 3px 9px;
             margin: 0 6px 8px 0; background: {SOFT}; border-radius: 4px;
             color: {INK}; }}
  .cc-tag b {{ font-weight: 600; color: {ACCENT}; margin-left: 6px; }}

  .cc-footer {{ display: flex; justify-content: space-between; gap: 12px;
                flex-wrap: wrap; font-size: 12px; color: {MUTED};
                border-top: 1px solid {LINE}; margin-top: 36px;
                padding-top: 12px; }}

  @media (max-width: 760px) {{
    .stApp {{ background: linear-gradient(180deg, {BAND} 0,
              {BAND} {band_px + 170}px, {PAGE} {band_px + 170}px) no-repeat,
              {PAGE}; }}
    .stTabs [role="tablist"] {{ position: static; margin: 8px 0 0; }}
    .cc-h1 {{ font-size: 24px; margin-top: 22px; }}
  }}
</style>
"""


def top_bar(area: str = "", right: str = "") -> str:
    area_html = f'<span class="cc-area">{area}</span>' if area else ""
    right_html = f'<span class="cc-env">{right}</span>' if right else ""
    return (f'<div class="cc-top">{LOGO}<span class="cc-word">ClariCase</span>'
            f'{area_html}{right_html}</div>')


def stat(label: str, value: str, foot: str = "") -> str:
    return (f'<div class="cc-stat"><div class="cc-stat-label">{label}</div>'
            f'<div class="cc-stat-num">{value}</div>'
            f'<div class="cc-stat-foot">{foot}</div></div>')


def gauge(conf: float, color: str = ACCENT) -> str:
    """Ring showing confidence; the circumference of r=15.5 is ~97.4."""
    pct = round(min(max(conf, 0.0), 1.0) * 100)
    arc = pct / 100 * 97.4
    return (f'<div class="cc-gauge" role="img" aria-label="Confidence {pct} '
            f'percent"><svg viewBox="0 0 36 36"><circle cx="18" cy="18" r="15.5"'
            f' fill="none" stroke="#fff" stroke-width="3.5"/><circle cx="18" '
            f'cy="18" r="15.5" fill="none" stroke="{color}" stroke-width="3.5" '
            f'stroke-linecap="round" stroke-dasharray="{arc:.1f} 97.4"/></svg>'
            f'<div class="cc-gauge-v"><div><b>{pct}%</b><small>confidence'
            f'</small></div></div></div>')


def team_code(team_id: str | None, team_name: str) -> str:
    if team_id in TEAM_CODES:
        return TEAM_CODES[team_id]
    words = [w for w in team_name.split() if w.isalpha() and w != "Team"]
    return "".join(w[0] for w in words[:2]).upper() or "?"


def esc(text: str) -> str:
    return _html.escape(text)
