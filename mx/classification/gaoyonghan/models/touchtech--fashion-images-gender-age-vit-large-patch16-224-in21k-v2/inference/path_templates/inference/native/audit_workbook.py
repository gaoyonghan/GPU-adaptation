import openpyxl,sys,math,json
source='__MODEL_ROOT__/resources/runtime/assets/workspace/9.11 100个模型结果汇总_五维度.xlsx'
out='/home/ubuntu/Documents/Codex/2026-09-27/codex-threads-01a0de0e-9d72-7fc0-b485/outputs/100个模型结果汇总_替代集补齐.xlsx'
a=openpyxl.load_workbook(source,data_only=False); z=openpyxl.load_workbook(out,data_only=False)
assert a.sheetnames==z.sheetnames,['sheetnames',a.sheetnames,z.sheetnames]
s=a.active;t=z.active
assert (s.max_row,s.max_column)==(t.max_row,t.max_column)
allowed={'A3','A6','C84','D84','C93','D93','C94','D94','C98','D98','C108','D108'}
changes=[];unintended=[];style_changes=[]
for row in s:
 for c in row:
  d=t[c.coordinate]
  if c.value!=d.value:
   (changes if c.coordinate in allowed else unintended).append((c.coordinate,str(c.value)[:80],str(d.value)[:80]))
  if c.coordinate not in allowed and (c.number_format!=d.number_format or c.fill.patternType!=d.fill.patternType or c.fill.fgColor!=d.fill.fgColor or (c.font.name,c.font.sz,c.font.b,c.font.i,c.font.color)!=(d.font.name,d.font.sz,d.font.b,d.font.i,d.font.color) or str(c.alignment)!=str(d.alignment)):style_changes.append(c.coordinate)
assert not unintended,['unintended_changes',unintended[:20]]
assert set(x[0] for x in changes)==allowed,['missing_changes',changes]
assert not style_changes,['visible_style_changes',style_changes[:20]]
assert all((i==98 and t.row_dimensions[i].height==72) or s.row_dimensions[i].height==t.row_dimensions[i].height for i in range(1,109))
assert all(abs(s.column_dimensions[x].width-t.column_dimensions[x].width)<0.001 for x in 'ABCDEF')
for row in [84,93,94,98,108]:
 assert '非官方可比' in t[f'C{row}'].value
 assert 'CPU前向' in t[f'D{row}'].value
 print(row,t[f'B{row}'].value,t[f'C{row}'].value.replace('\n',' | '),t[f'D{row}'].value.replace('\n',' | '))
print('AUDIT_OK',len(changes),'intended changes; no other value/style changes')
