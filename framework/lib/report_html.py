"""Self-contained HTML report generation for BenchmarkResult."""

from __future__ import annotations

import json
from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .report import BenchmarkResult

TASK_LABELS: dict[str, str] = {
    "reading_comprehension": "Comprensión",
    "sentence_ordering": "Ord. oraciones",
    "sentence_elimination": "Eliminación",
    "verbal_series": "Series verbales",
    "analogies": "Analogías",
    "synonyms_and_antonyms": "Sin./Antón.",
    "incomplete_sentences": "Inc. oraciones",
}

METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "accuracy_drop": "Acc. Drop",
    "flip_rate": "Flip Rate",
    "consistency": "Consistency",
    "positive_transfer": "Pos. Transfer",
    "negative_transfer": "Neg. Transfer",
    "rank_consistency": "Rank Cons.",
}

_ROBUSTNESS_ATTRS = [
    "accuracy_drop",
    "flip_rate",
    "consistency",
    "positive_transfer",
    "negative_transfer",
    "rank_consistency",
]


def build_html(result: "BenchmarkResult") -> str:
    """Generate a self-contained HTML report from a BenchmarkResult."""
    result._compute_all_robustness()
    data = _build_compact_data(result)
    summary_html = _render_summary_table(result)
    return _render_page(data, summary_html, result)


# ---------------------------------------------------------------------------
# Data builder
# ---------------------------------------------------------------------------

def _collect_entities(
    result: "BenchmarkResult",
) -> tuple[list[str], list[str], list[str]]:
    """Return (models, attacks, tasks). Baseline is excluded from attacks (heatmap only shows attacked variants)."""
    models: list[str] = []
    attacks_seen: list[str] = []
    tasks: set[str] = set()

    for mr in result.models:
        if mr.model_name not in models:
            models.append(mr.model_name)
        for ds in mr.evaluated_datasets:
            lbl = ds.attack_label
            if lbl != "baseline" and lbl not in attacks_seen:
                attacks_seen.append(lbl)
            for t in ds.metrics.tasks:
                tasks.add(t)

    return models, sorted(attacks_seen), sorted(tasks)


def _detect_metrics(result: "BenchmarkResult", attacks: list[str]) -> list[str]:
    """Return only metrics that have at least one non-null value in the results."""
    metrics = ["accuracy"]
    if not any(a != "baseline" for a in attacks):
        return metrics

    for attr in _ROBUSTNESS_ATTRS:
        found = False
        for mr in result.models:
            for ds in mr.evaluated_datasets:
                if ds.attack is None or not ds._per_task_robustness:
                    continue
                for tr in ds._per_task_robustness.values():
                    if getattr(tr, attr, None) is not None:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            metrics.append(attr)

    return metrics


def _build_compact_data(result: "BenchmarkResult") -> dict:
    """Build the compact array-based data structure embedded in the HTML."""
    models, attacks, tasks = _collect_entities(result)
    metrics = _detect_metrics(result, attacks)

    # (model_name, attack_label) → DatasetResult
    lookup: dict[tuple[str, str], object] = {}
    for mr in result.models:
        for ds in mr.evaluated_datasets:
            lookup[(mr.model_name, ds.attack_label)] = ds

    def _get(ds, task: str, metric: str) -> float | None:
        if ds is None:
            return None
        if metric == "accuracy":
            info = ds.metrics.tasks.get(task)
            return round(info["accuracy"], 4) if info else None
        if ds.attack is None or not ds._per_task_robustness:
            return None
        tr = ds._per_task_robustness.get(task)
        if tr is None:
            return None
        val = getattr(tr, metric, None)
        return round(val, 4) if val is not None else None

    # d[model_idx][attack_idx][task_idx][metric_idx]
    d = [
        [
            [
                [_get(lookup.get((model, attack)), task, metric) for metric in metrics]
                for task in tasks
            ]
            for attack in attacks
        ]
        for model in models
    ]

    # avg[attack_idx][task_idx][metric_idx]
    nm = len(models)
    avg = []
    for ai in range(len(attacks)):
        a_avg = []
        for ti in range(len(tasks)):
            t_avg = []
            for ki in range(len(metrics)):
                vals = [d[mi][ai][ti][ki] for mi in range(nm) if d[mi][ai][ti][ki] is not None]
                t_avg.append(round(sum(vals) / len(vals), 4) if vals else None)
            a_avg.append(t_avg)
        avg.append(a_avg)

    # Human-readable attack labels
    attack_labels: dict[str, str] = {}
    for mr in result.models:
        for ds in mr.evaluated_datasets:
            if ds.attack is not None:
                lbl = ds.attack.label or ds.attack.attack_name
                attack_labels[lbl] = lbl.replace("_", " ").title()

    # total_samples from baseline
    total_samples = 0
    for mr in result.models:
        for ds in mr.evaluated_datasets:
            if ds.attack is None:
                total_samples = ds.metrics.total
                break
        if total_samples:
            break

    return {
        "info": {
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "is_finished": result.is_finished,
            "baseline": result.baseline_file,
            "total_samples": total_samples,
        },
        "models": models,
        "attacks": attacks,
        "tasks": tasks,
        "metrics": metrics,
        "metric_labels": {m: METRIC_LABELS.get(m, m) for m in metrics},
        "task_labels": {t: TASK_LABELS.get(t, t) for t in tasks},
        "attack_labels": attack_labels,
        "d": d,
        "avg": avg,
    }


# ---------------------------------------------------------------------------
# Summary table (static HTML, no JS)
# ---------------------------------------------------------------------------

def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.2%}"


def _fmt_ms(v: float) -> str:
    return f"{v:,.0f}"


def _render_summary_table(result: "BenchmarkResult") -> str:
    has_attacks = any(
        ds.attack is not None
        for mr in result.models
        for ds in mr.evaluated_datasets
    )
    show_rank = any(
        ds._robustness is not None and ds._robustness.rank_consistency is not None
        for mr in result.models
        for ds in mr.evaluated_datasets
    )

    rob_cols: list[str] = []
    if has_attacks:
        rob_cols = [
            "accuracy_drop", "flip_rate", "consistency",
            "positive_transfer", "negative_transfer",
        ]
        if show_rank:
            rob_cols.append("rank_consistency")

    lines: list[str] = ['<table class="summary-table">']
    lines.append("<thead><tr>")
    lines.append("<th>Modelo</th><th>Dataset / Ataque</th>")
    lines.append("<th>Total</th><th>Correctas</th><th>Fallos</th>")
    lines.append("<th>Accuracy</th><th>Lat. prom. (ms)</th>")
    for col in rob_cols:
        lbl = METRIC_LABELS.get(col, col)
        title = ""
        if col == "rank_consistency":
            title = ' title="Spearman ρ entre logprobs. Requiere logprobs=True."'
        lines.append(f"<th{title}>{escape(lbl)}</th>")
    lines.append("</tr></thead><tbody>")

    for mr in result.models:
        for ds in mr.evaluated_datasets:
            m = ds.metrics
            r = ds._robustness
            failed_cls = ' class="cell-error"' if m.failed > 0 else ""
            atk_display = (
                "Baseline"
                if ds.attack is None
                else (ds.attack.label or ds.attack.attack_name).replace("_", " ").title()
            )
            lines.append("<tr>")
            lines.append(f'<td class="model-name">{escape(mr.model_name)}</td>')
            lines.append(f"<td>{escape(atk_display)}</td>")
            lines.append(f"<td>{m.total}</td>")
            lines.append(f"<td>{m.correct}</td>")
            lines.append(f"<td{failed_cls}>{m.failed}</td>")
            lines.append(f"<td>{_fmt_pct(m.accuracy)}</td>")
            lines.append(f"<td>{_fmt_ms(m.avg_latency_ms)}</td>")
            for col in rob_cols:
                if r is None:
                    lines.append('<td class="cell-na">—</td>')
                else:
                    val = getattr(r, col, None)
                    lines.append(f"<td>{_fmt_pct(val)}</td>")
            lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------

def _render_page(data: dict, summary_html: str, result: "BenchmarkResult") -> str:
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    status_label = "Completado" if result.is_finished else "Parcial"
    status_cls = "status-ok" if result.is_finished else "status-partial"

    started = result.started_at[:19].replace("T", " ") if result.started_at else "—"
    finished = result.finished_at[:19].replace("T", " ") if result.finished_at else "—"

    return (
        _HTML_TEMPLATE
        .replace("/*__DATA__*/", data_json)
        .replace("<!--SUMMARY-->", summary_html)
        .replace("<!--STATUS_LABEL-->", escape(status_label))
        .replace("<!--STATUS_CLS-->", status_cls)
        .replace("<!--STARTED_AT-->", escape(started))
        .replace("<!--FINISHED_AT-->", escape(finished))
        .replace("<!--BASELINE-->", escape(result.baseline_file))
    )


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Verbal Reasoning &mdash; Evaluation Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f0f;color:#e0e0e0;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:14px;line-height:1.5;padding:28px 32px}
h1{font-size:22px;font-weight:600;color:#fff;margin-bottom:6px}
h2{font-size:11px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:16px}
section{margin-bottom:48px}
.info-bar{display:flex;flex-wrap:wrap;gap:20px;margin-bottom:36px;font-size:13px;color:#666}
.info-bar span strong{color:#999;margin-right:4px}
.status-ok{color:#00c870;font-weight:500}
.status-partial{color:#f0a030;font-weight:500}

/* Summary table */
.summary-table{border-collapse:collapse;width:100%;font-size:13px}
.summary-table th{background:#161616;color:#666;font-weight:500;text-align:left;padding:8px 14px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #222;white-space:nowrap;cursor:default}
.summary-table th[title]{border-bottom:1px dashed #444;cursor:help}
.summary-table td{padding:8px 14px;border-bottom:1px solid #181818;color:#ccc;white-space:nowrap}
.summary-table tbody tr:hover td{background:#141414}
.summary-table .model-name{color:#fff;font-weight:500}
.summary-table .cell-error{color:#f05555;font-weight:600}
.summary-table .cell-na{color:#333}

/* Heatmap controls */
#controls{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;gap:12px;flex-wrap:wrap}
.ctrl-left{display:flex;align-items:center;gap:8px}
.ctrl-left label{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.08em}
#secondary-sel{background:#161616;color:#ccc;border:1px solid #2a2a2a;border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer;outline:none;appearance:none;-webkit-appearance:none;padding-right:24px;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23666'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 8px center}
#secondary-sel:hover{border-color:#444}
#secondary-sel:focus{border-color:#555}
#swap-btn{background:#161616;color:#777;border:1px solid #2a2a2a;border-radius:6px;padding:6px 11px;font-size:16px;cursor:pointer;line-height:1;transition:color .15s,border-color .15s}
#swap-btn:hover{color:#ddd;border-color:#444}
#metric-btns{display:flex;gap:4px;flex-wrap:wrap}
.metric-btn{background:transparent;color:#666;border:1px solid #2a2a2a;border-radius:6px;padding:5px 13px;font-size:12px;cursor:pointer;transition:all .15s;white-space:nowrap}
.metric-btn:hover{color:#bbb;border-color:#444}
.metric-btn.active{background:#ffffff;color:#000;border-color:#fff;font-weight:600}

/* Heatmap table */
#table-wrap{overflow-x:auto}
#table-wrap table{border-collapse:collapse;min-width:100%}
#table-wrap thead th{color:#666;font-weight:500;padding:10px 18px;text-align:center;font-size:12px;white-space:nowrap;user-select:none;border-bottom:1px solid #1e1e1e}
#table-wrap thead th:first-child{text-align:left;min-width:180px;cursor:default}
#table-wrap thead th[data-ci]{cursor:pointer}
#table-wrap thead th[data-ci]:hover{color:#ccc}
#table-wrap tbody td{text-align:center;padding:10px 18px;color:#fff;font-size:13px;font-weight:500;border-bottom:1px solid #0f0f0f}
#table-wrap tbody td:first-child{text-align:left;color:#ccc;font-weight:400;font-size:13px;background:#0f0f0f!important;padding-left:0}
#table-wrap .null-cell{background:#141414!important;color:#2e2e2e!important;font-weight:400}
#table-wrap .avg-row td{border-top:1px solid #1e1e1e;border-bottom:none}
#table-wrap .avg-row td:first-child{color:#555;font-style:italic}
</style>
</head>
<body>

<h1>LLM Verbal Reasoning &mdash; Evaluation Report</h1>
<div class="info-bar">
  <span><strong>Inicio:</strong><!--STARTED_AT--></span>
  <span><strong>Fin:</strong><!--FINISHED_AT--></span>
  <span><strong>Dataset:</strong><!--BASELINE--></span>
  <span><strong>Estado:</strong><span class="<!--STATUS_CLS-->"><!--STATUS_LABEL--></span></span>
</div>

<section>
<h2>Resumen general</h2>
<!--SUMMARY-->
</section>

<section>
<h2>An&#225;lisis comparativo</h2>
<div id="controls">
  <div class="ctrl-left">
    <label id="sel-label">Ataque</label>
    <select id="secondary-sel"></select>
    <button id="swap-btn" title="Intercambiar ejes">&#8646;</button>
  </div>
  <div id="metric-btns"></div>
</div>
<div id="table-wrap"></div>
</section>

<script>
const DATA=/*__DATA__*/;
const S={mode:"task_primary",mIdx:0,sIdx:0,sortCol:null,sortDir:-1};

function gv(mi,ci){return S.mode==="task_primary"?DATA.d[mi][S.sIdx][ci][S.mIdx]:DATA.d[mi][ci][S.sIdx][S.mIdx];}
function ga(ci){return S.mode==="task_primary"?DATA.avg[S.sIdx][ci][S.mIdx]:DATA.avg[ci][S.sIdx][S.mIdx];}
function cols(){return S.mode==="task_primary"?DATA.tasks:DATA.attacks;}
function colLbl(c){return S.mode==="task_primary"?(DATA.task_labels[c]||c):(DATA.attack_labels[c]||c);}
function fmt(v){return v===null?"—":v.toFixed(2);}

function cellBg(v,mn,mx){
  if(v===null)return"#141414";
  const t=mx===mn?0.5:(v-mn)/(mx-mn);
  const sat=Math.round(70-50*t);
  const lgt=Math.round(40-32*t);
  return"hsl(152,"+sat+"%,"+lgt+"%)";
}

function render(){
  const cs=cols(),nc=cs.length,nm=DATA.models.length;
  const mins=[],maxs=[];
  for(let ci=0;ci<nc;ci++){
    const vs=[];
    for(let mi=0;mi<nm;mi++){const v=gv(mi,ci);if(v!==null)vs.push(v);}
    const av=ga(ci);if(av!==null)vs.push(av);
    mins[ci]=vs.length?Math.min(...vs):0;
    maxs[ci]=vs.length?Math.max(...vs):1;
  }

  const ord=DATA.models.map((_,i)=>i);
  if(S.sortCol!==null){
    ord.sort((a,b)=>{
      const va=gv(a,S.sortCol),vb=gv(b,S.sortCol);
      if(va===null&&vb===null)return 0;
      if(va===null)return 1;
      if(vb===null)return -1;
      return S.sortDir*(va-vb);
    });
  }

  let h="<table><thead><tr><th></th>";
  for(let ci=0;ci<nc;ci++){
    const c=cs[ci];
    const arr=S.sortCol===ci?(S.sortDir===1?" ↑":" ↓"):"";
    h+="<th data-ci='"+ci+"'>"+colLbl(c)+arr+"</th>";
  }
  h+="</tr></thead><tbody>";

  for(const mi of ord){
    h+="<tr><td class='model-name'>"+DATA.models[mi]+"</td>";
    for(let ci=0;ci<nc;ci++){
      const v=gv(mi,ci);
      const bg=cellBg(v,mins[ci],maxs[ci]);
      h+="<td style='background:"+bg+"'"+(v===null?" class='null-cell'":"")+">"+ fmt(v)+"</td>";
    }
    h+="</tr>";
  }

  h+="<tr class='avg-row'><td>Promedio</td>";
  for(let ci=0;ci<nc;ci++){
    const v=ga(ci);
    const bg=cellBg(v,mins[ci],maxs[ci]);
    h+="<td style='background:"+bg+"'"+(v===null?" class='null-cell'":"")+">"+ fmt(v)+"</td>";
  }
  h+="</tr></tbody></table>";

  const wrap=document.getElementById("table-wrap");
  wrap.innerHTML=h;
  wrap.querySelectorAll("th[data-ci]").forEach(function(th){
    th.addEventListener("click",function(){
      const ci=+th.dataset.ci;
      if(S.sortCol===ci){S.sortDir*=-1;}else{S.sortCol=ci;S.sortDir=-1;}
      render();
    });
  });
}

function syncSel(){
  const items=S.mode==="task_primary"?DATA.attacks:DATA.tasks;
  const lbls=S.mode==="task_primary"?DATA.attack_labels:DATA.task_labels;
  const sel=document.getElementById("secondary-sel");
  sel.innerHTML=items.map(function(x,i){return"<option value='"+i+"'>"+(lbls[x]||x)+"</option>";}).join("");
  sel.value=S.sIdx;
  document.getElementById("sel-label").textContent=S.mode==="task_primary"?"Ataque":"Tarea";
}

const mb=document.getElementById("metric-btns");
mb.innerHTML=DATA.metrics.map(function(m,i){
  return"<button class='metric-btn"+(i===0?" active":"")+"' data-i='"+i+"'>"+DATA.metric_labels[m]+"</button>";
}).join("");
mb.querySelectorAll(".metric-btn").forEach(function(b){
  b.addEventListener("click",function(){
    mb.querySelectorAll(".metric-btn").forEach(function(x){x.classList.remove("active");});
    b.classList.add("active");
    S.mIdx=+b.dataset.i;
    render();
  });
});

document.getElementById("secondary-sel").addEventListener("change",function(e){S.sIdx=+e.target.value;render();});
document.getElementById("swap-btn").addEventListener("click",function(){
  S.mode=S.mode==="task_primary"?"attack_primary":"task_primary";
  S.sIdx=0;S.sortCol=null;
  syncSel();render();
});

syncSel();render();
</script>
</body>
</html>"""
