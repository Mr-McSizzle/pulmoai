import type { PatientTimeline, TimelineView } from "../types";
import { ArrowRight, Download, PlayCircle } from "lucide-react";
const steps: {view:TimelineView; title:string; description:string}[] = [
  {view:"visits",title:"1. Show the visits",description:"Four checked visits over six months, plus one report ready for live review."},
  {view:"labs",title:"2. Compare lab evolution",description:"Cholesterol and hemoglobin approach their intervals; creatinine moves away."},
  {view:"risks",title:"3. Show calculated risk changes",description:"Select an outcome and horizon. Actual installed models calculate the estimates."},
  {view:"images",title:"4. Compare source images",description:"Earlier/later generated illustrations demonstrate archive comparison. No AI scores."},
  {view:"rna",title:"5. Inspect RNA history",description:"Repeated generated profile, dated results and reported-sex model context."},
  {view:"assessment",title:"6. Open the overall assessment",description:"Combined findings, review priorities and explicit coverage gaps."},
];
export function PresentationCaseGuide({timeline, onView, onPractice}: {timeline:PatientTimeline; onView:(view:TimelineView)=>void; onPractice:()=>void}) {
  const example = timeline.example_case;
  if (!example) return null;
  const practice = timeline.records.find(record=>record.id===example.practice_record_id);
  return <section className="content-panel presentation-guide" aria-label="Presentation case guide">
    <header><div><p className="eyebrow">READY TO WALK THROUGH</p><h2>{example.title}</h2><p>{example.story}</p></div><span className="data-origin-badge">Generated example records</span></header>
    <p className="timeline-notice">{example.validation_notice}</p>
    <div className="presentation-steps">{steps.map(step=><button key={step.view} onClick={()=>onView(step.view)}><PlayCircle size={17}/><strong>{step.title}</strong><span>{step.description}</span><ArrowRight size={15}/></button>)}</div>
    <footer><button className="button-primary" disabled={!practice} onClick={onPractice}>{practice?.review_status==="reviewed"?"Reopen the source-review exercise":"Start live source-review exercise"}<ArrowRight size={15}/></button><a className="button-secondary" href={`/api/patients/${timeline.patient.id}/example-pack`} download><Download size={15}/>Download all example sources</a><span>Reopening this case preserves your reviews and edits.</span></footer>
    <details><summary>Processing availability and example provenance</summary><ul>{example.notices.map(note=><li key={note}>{note}</li>)}</ul></details>
  </section>;
}
