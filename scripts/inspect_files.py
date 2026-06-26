import pandas as pd
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

base = r"E:\Langlobal\AIPEMVP_0526"
files = ['CQA库0526.xlsx', 'Designer库0526.xlsx', 'GT库0526.xlsx', 'LQA库0526.xlsx', 'Stable库0526.xlsx']

out_lines = []
for f in files:
    p = os.path.join(base, f)
    out_lines.append('='*70)
    out_lines.append(f'文件: {f}  大小: {os.path.getsize(p)/1024/1024:.1f}MB')
    try:
        xl = pd.ExcelFile(p)
        out_lines.append(f'Sheets: {xl.sheet_names}')
        for s in xl.sheet_names:
            df = pd.read_excel(p, sheet_name=s)
            out_lines.append(f'\n  [Sheet={s}]  shape={df.shape}')
            out_lines.append(f'  Columns: {list(df.columns)}')
            out_lines.append('  Head:')
            head = df.head(3).to_string(max_colwidth=80)
            for line in head.split('\n'):
                out_lines.append('    ' + line)
    except Exception as e:
        out_lines.append(f'  ERROR: {e}')

content = '\n'.join(out_lines)
print(content)
with open(r'E:\Langlobal\AIPEMVP_0526\inspect_files.txt', 'w', encoding='utf-8') as fp:
    fp.write(content)
