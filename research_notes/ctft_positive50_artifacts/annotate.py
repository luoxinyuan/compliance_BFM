from pathlib import Path
import json,subprocess
s=Path(__file__).resolve().parent
v=Path('/Users/gaoe/Projects/compliance-bfm-diagnostics/videos')
out=v/'CTFT-positive50';out.mkdir(exist_ok=True)
x=json.loads((s/'video_channels.json').read_text())
for tag,src,layout in [('01','02_native_feedback_comparison',[('TL C32','C32__speed0.15'),('TR Native','motion__speed0.15'),('BL Rolling','rolling_continue'),('BR I','I__speed0.15')]),('02','03_common_prefix_stop_continue',[('Top: continue','rolling_continue'),('Bottom: stop','rolling_stop')])]:
 lines=[]
 def clock(k):return '0:%02d:%05.2f'%(int(k//60),k%60)
 for frame in range(320):
  t=3+frame/25;text=[]
  for label,n in layout:
   rows=x[n];j=round((t-4)/.04)
   if t<4:value='source'
   elif t>rows[-1]['t']+.035:value='ended'
   else:
    row=rows[min(max(j,0),len(rows)-1)]
    value=('grasp' if row['active'] else 'released' if t>5 else 'pre')+' %.1f N'%row['f']+' | '+row['stance']
   text.append(label+' '+value)
  lines.append('Dialogue: 0,'+clock(frame/25)+','+clock((frame+1)/25)+',Default,,0,0,0,,'+'    '.join(text[:2])+r'\N'+'    '.join(text[2:])+r'\N'+'Stance OK / FAIL = whole loaded segment; - = no qualified segment. Offline score.')
 ass=s/(tag+'.ass')
 ass.write_text('[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 1392\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,25,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,8,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'+'\n'.join(lines))
 dest=out/(src+'_stance.mp4')
 subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(v/'native-rolling-feedback'/(src+'.mp4')),'-vf',f'pad=iw:ih+112:0:0,ass={ass}','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
