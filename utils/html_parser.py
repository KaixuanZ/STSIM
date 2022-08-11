from html_table_parser.parser import HTMLTableParser

with open('../data/MacroTexture3K_pappas_group.txt') as f:
    lines = f.readlines()

html_text = ''.join(lines)

p = HTMLTableParser()
p.feed(html_text)
print(p.tables)

table = p.tables[0]

groups = {}
# reverse order
excludes = (40-21,40-27,40-28,40-29,40-30,40-31,40-32,40-33,40-34,40-35,40-36,40-37,40-38,40-39)
for row in table[1:]:
    idx = len(table) - int(row[0])
    if idx not in excludes:
        groups[idx] = row[1:]

import pdb;pdb.set_trace()
