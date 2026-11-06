import numpy as np,json,sys
X=sys.argv[1];W='.work/model-qnn-meta-20261030/full-validation/primary/'
V=json.load(open('.work/model-qnn-meta-20261030/full-validation.json'))['models']['primary']['runs']
def m(a,b):
    d=a.astype('f8')-b.astype('f8');e=np.abs(d)
    return dict(mae=float(e.mean()),bias=float(d.mean()),p95=float(np.percentile(e,95)),p99=float(np.percentile(e,99)),max=float(e.max()),diff=float((e>0).mean()))
R={}
for r in V:
    c=r['case'];e8=np.fromfile(W+c+'/cpu-u8.raw','u1');ef=np.fromfile(W+c+'/cpu-f32.raw','<f4')
    q8=np.fromfile(W+c+'/u8.raw','u1');qf=np.fromfile(W+c+'/f32.raw','<f4')
    row={'model':{'f32':m(qf,ef),'u8':m(q8,e8)},'ms':[x['ms'] for x in r['calls']]}
    row['control_equal']=bool(np.array_equal(np.fromfile(f'{X}/out/emu/{c}/out1.raw','<f4'),ef) and np.array_equal(np.fromfile(f'{X}/out/emu/{c}/out0.raw','u1'),e8))
    for b in('hw','fix'):
        h8=np.fromfile(f'{X}/out/{b}/{c}/out0.raw','u1');hf=np.fromfile(f'{X}/out/{b}/{c}/out1.raw','<f4')
        row[b]={'f32':m(hf,ef),'u8':m(h8,e8),'f32_vs_model':m(hf,qf),'u8_vs_model':m(h8,q8)}
    R[c]=row
    Q,H,F=row['model']['f32'],row['hw']['f32'],row['fix']['f32']
    ok=lambda A:'ok' if A['mae']<=Q['mae'] and abs(A['bias'])<=abs(Q['bias']) and A['p95']<=Q['p95'] else 'NO'
    print(f"{c:14s}|Q {Q['mae']:.5f} {Q['bias']:+.5f} {Q['p95']:.4f} {Q['max']:.3f} |H {H['mae']:.4f} {H['bias']:+.4f} {H['p95']:.3f} {H['max']:.2f} |F {F['mae']:.4f} {F['bias']:+.4f} {F['p95']:.3f} {F['max']:.2f} |u8 {row['model']['u8']['mae']:.3f}/{row['hw']['u8']['mae']:.3f}/{row['fix']['u8']['mae']:.3f} |{ok(H)} {ok(F)} ctl={int(row['control_equal'])}")
json.dump(R,open(f'{X}/metrics.json','w'),indent=1)
med=lambda k,f='mae':float(np.median([abs(R[c][k]['f32'][f]) for c in R]))
for f in('mae','bias','p95','max'): print(f,'median Q/HW/FIX',round(med('model',f),5),round(med('hw',f),5),round(med('fix',f),5))
print('control equal',sum(R[c]['control_equal'] for c in R),'/',len(R))
for b in('hw','fix'):print(b,'cases with mae<=Model',sum(R[c][b]['f32']['mae']<=R[c]['model']['f32']['mae'] for c in R),'bias<=',sum(abs(R[c][b]['f32']['bias'])<=abs(R[c]['model']['f32']['bias']) for c in R),'p95<=',sum(R[c][b]['f32']['p95']<=R[c]['model']['f32']['p95'] for c in R),'max<=',sum(R[c][b]['f32']['max']<=R[c]['model']['f32']['max'] for c in R))
