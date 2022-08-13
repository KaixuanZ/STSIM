
from html_table_parser.parser import HTMLTableParser

with open('../data/MacroTexture3K_pappas_group.txt') as f:
    lines = f.readlines()

html_text = ''.join(lines)

p = HTMLTableParser()
p.feed(html_text)

table = p.tables[0]

imgnames = {}
groups = {}
# reverse order
excludes = (40-21,40-27,40-28,40-29,40-30,40-31,40-32,40-33,40-34,40-35,40-36,40-37,40-38,40-39)
for row in table[1:]:
    idx = len(table) - int(row[0])
    if idx not in excludes:
        groups[idx] = [f[:-7]+'.png' for f in row[1:]]
        for f in row[1:]:
            if f[:-7]+'.png' not in imgnames:
                imgnames[f[:-7]+'.png'] = [idx]
            else:
                imgnames[f[:-7]+'.png'].append(idx)

for key in imgnames:
    imgnames[key] = list(set(imgnames[key]))
output_path = 'grouping_sets.json'

import json
with open(output_path, 'w') as json_file:
    json.dump(groups, json_file)

output_path = 'grouping_imgnames.json'
with open(output_path, 'w') as json_file:
    json.dump(imgnames, json_file)
import pdb;pdb.set_trace()

