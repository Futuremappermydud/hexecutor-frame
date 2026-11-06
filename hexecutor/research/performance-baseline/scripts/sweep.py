import json,os,subprocess,hashlib,sys
X,R=sys.argv[1],os.getcwd()
V=json.load(open('.work/model-qnn-meta-20261030/full-validation.json'))['models']['primary']['runs']
bad=0
for r in V:
    c=r['case']
    d={'smoke':'frame/build/native-smoke','random':'frame/build/depth-inputs/random'}.get(c,f'{X}/inputs/{c}')
    for k,h in r['input_sha256'].items():
        if hashlib.sha256(open(f'{d}/{k}.raw','rb').read()).hexdigest()!=h: print('INPUT MISMATCH',c,k);bad+=1
    if bad: continue
    for b in sys.argv[2:]:
        o=f'{X}/out/{b}/{c}';os.makedirs(o,exist_ok=True)
        if os.path.exists(o+'/out1.raw'):continue
        p=subprocess.run([f'{X}/bin/test_{b}','primary','reference','whole'],env=dict(os.environ,QD_EXPECT='print',QD_DUMP=o,QD_INPUTS=d),capture_output=True,text=True)
        if p.returncode: print('FAIL',b,c,p.stderr[-200:])
print('done bad',bad)
