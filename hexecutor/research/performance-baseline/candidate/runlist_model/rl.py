import sys,re
sys.path.insert(0,'/tmp/frame-i-merge/depth-model/frame/native_program')
import compact_v2 as pp
R='/tmp/frame-i-merge/depth-model/frame/native_program/'
prog=pp.decode(open(R+'examples/primary-v2.nprg','rb').read())
cmds=prog['commands']
h=open(R+'qnn_runlist_primary.generated.h').read()
def arr(name):
    m=re.search(r'%s\[[^\]]*\]\s*=\s*\{(.*?)\};'%name,h,re.S); return m.group(1)
def entries(name):
    body=arr(name); return [tuple(int(x,0) for x in e.split(',')) for e in re.findall(r'\{([^{}]*)\}',body)]
main=entries('nprg2_qnn_primary_main'); work=entries('nprg2_qnn_primary_worker')
owner=[int(x) for x in re.findall(r'\d+',arr('nprg2_qnn_primary_cp_owner'))]
init=[int(x) for x in re.findall(r'\d+',arr('nprg2_qnn_primary_cp_initial'))]
NAMES={1:'NOP',2:'INSLICEPAD',3:'INSLICE',4:'W2VTCM',5:'IM2COL',6:'CONV',7:'S2D',8:'REQ8',9:'REQ16',10:'ADD8',11:'ADD16',12:'ADDSUB16',13:'ADD1111',14:'MUL16',15:'MUL1111',16:'COPYV',17:'COPYA',18:'TRANSP',19:'CONCAT',20:'GATHER',21:'SLICE',22:'SLICEPAD',23:'DECIM',24:'RESIZE',25:'D2S',26:'SPILL',27:'OUTSLICE',28:'QUANT',29:'DEQ',30:'SIGN',31:'MEAN',32:'WSCALES',33:'INVSCALE',34:'COMBSC',35:'CONVW',36:'SCALECONV',37:'DYNBIAS',38:'BIASSHUF',39:'LINCLIP',40:'ABS',41:'MULINL',42:'ADDSUBFL',43:'ADDSUBIN',44:'ZAP'}
def name(ix):
    c=cmds[ix];return '%s/%d'%(NAMES.get(c[1],c[1]),c[2])
