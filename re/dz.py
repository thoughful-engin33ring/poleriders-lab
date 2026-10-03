import os, re, struct, sys
S=os.path.dirname(os.path.abspath(__file__))
b=open(os.path.expanduser('~/Library/Application Support/Steam/steamapps/common/Sportsfriends/Sportsfriends.app/Contents/MacOS/Sportsfriends.orig'),'rb').read()
L=open(S+'/dis.txt').read().split('\n')
rx=re.compile(r'^([0-9a-f]+):\t(.*)$')
rows=[(int(m.group(1),16),m.group(2)) for m in map(rx.match,L) if m]
start=int(sys.argv[1],16); cnt=int(sys.argv[2])
i=next(k for k,(a,_) in enumerate(rows) if a>=start)
for j in range(i,i+cnt):
    a,ins=rows[j]; r=re.search(r'(-?0x[0-9a-f]+)\(%rip\)',ins); x=''
    if r:
        t=rows[j+1][0]+int(r.group(1),16); o=t-0x100000000
        if 0<o<len(b)-8:
            op=ins.split('\t')[0]
            x=f' ;{struct.unpack("<d",b[o:o+8])[0]:g}' if op.endswith('sd') else f' ;{struct.unpack("<f",b[o:o+4])[0]:g}'
    print(hex(a),ins[:100],x)
