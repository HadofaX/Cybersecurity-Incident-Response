import math, random, csv, json, statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'csv'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
REPEATS = 5
GAMMA = 0.90
MAX_ALLOWED_STEP = 5
random.seed(SEED)

states = ['S0','S1','S2','S3','S4']
active_states = ['S0','S1','S2','S3']
state_name = {
    'S0':'Normal','S1':'Kimlik ele gecirme','S2':'Yanal hareket','S3':'Veri sizdirma','S4':'Sinirlandirilmis'
}
harm = {'S0':0,'S1':25,'S2':60,'S3':100,'S4':-20}
prior = {'S0':0.55,'S1':0.25,'S2':0.15,'S3':0.05,'S4':0.0}

obs_prob = {
    'S0': {'O0':0.70,'O1':0.20,'O2':0.08,'O3':0.02},
    'S1': {'O0':0.15,'O1':0.65,'O2':0.15,'O3':0.05},
    'S2': {'O0':0.08,'O1':0.20,'O2':0.60,'O3':0.12},
    'S3': {'O0':0.03,'O1':0.07,'O2':0.20,'O3':0.70},
    'S4': {'O0':0.75,'O1':0.10,'O2':0.10,'O3':0.05},
}
deep_obs_prob = {
    'S0': {'O0':0.82,'O1':0.10,'O2':0.05,'O3':0.02,'O4':0.01},
    'S1': {'O0':0.05,'O1':0.70,'O2':0.10,'O3':0.05,'O4':0.10},
    'S2': {'O0':0.03,'O1':0.08,'O2':0.68,'O3':0.06,'O4':0.15},
    'S3': {'O0':0.01,'O1':0.03,'O2':0.10,'O3':0.71,'O4':0.15},
    'S4': {'O0':0.85,'O1':0.05,'O2':0.04,'O3':0.01,'O4':0.05},
}

actions = ['A0','A1','A2','A3','A4','A5']
action_name = {
    'A0':'Izlemeye devam et',
    'A1':'Ek log ve EDR incelemesi yap',
    'A2':'MFA dogrulamasi iste',
    'A3':'Supheli oturumu sonlandir ve hesabi sifirla',
    'A4':'Uc cihazi izole et',
    'A5':'Insan guvenlik uzmanina yonlendir',
}
action_cost = {'A0':0,'A1':5,'A2':7,'A3':15,'A4':30,'A5':12}
normal_penalty = {'A0':0,'A1':2,'A2':4,'A3':20,'A4':55,'A5':8}
contain = {
    'A0':{'S0':0.00,'S1':0.05,'S2':0.02,'S3':0.01},
    'A1':{'S0':0.00,'S1':0.10,'S2':0.05,'S3':0.03},
    'A2':{'S0':0.00,'S1':0.55,'S2':0.15,'S3':0.05},
    'A3':{'S0':0.00,'S1':0.80,'S2':0.35,'S3':0.10},
    'A4':{'S0':0.00,'S1':0.50,'S2':0.80,'S3':0.65},
    'A5':{'S0':0.00,'S1':0.65,'S2':0.65,'S3':0.50},
}

T = {
    'S0': {'S0':0.94,'S1':0.05,'S2':0.01,'S3':0.00,'S4':0.00},
    'S1': {'S0':0.05,'S1':0.68,'S2':0.22,'S3':0.05,'S4':0.00},
    'S2': {'S0':0.00,'S1':0.05,'S2':0.65,'S3':0.25,'S4':0.05},
    'S3': {'S0':0.00,'S1':0.00,'S2':0.05,'S3':0.90,'S4':0.05},
    'S4': {'S0':0.05,'S1':0.00,'S2':0.00,'S3':0.00,'S4':0.95},
}

tests = {
    'T1': {'name':'Ayrintili kimlik dogrulama logu','cost':3,'sens':0.78,'spec':0.82},
    'T2': {'name':'MFA dogrulama testi','cost':5,'sens':0.85,'spec':0.90},
    'T3': {'name':'Derin EDR taramasi','cost':8,'sens':0.90,'spec':0.88},
    'T4': {'name':'Ag trafik incelemesi','cost':7,'sens':0.88,'spec':0.85},
    'T5': {'name':'Insan uzman dogrulamasi','cost':12,'sens':0.95,'spec':0.93},
}

scenarios = {
    'S1':  {'obs':['O0','O0'], 'true':'S0', 'feature':'Normal kullanici davranisi'},
    'S2':  {'obs':['O1','O1'], 'true':'S1', 'feature':'Tekrarlayan kimlik dogrulama anomalisi'},
    'S3':  {'obs':['O1','O2'], 'true':'S2', 'feature':'Kimlik anomalisi sonrasi uc nokta alarmi'},
    'S4':  {'obs':['O2','O3'], 'true':'S3', 'feature':'Yanal hareket sonrasi veri cikisi'},
    'S5':  {'obs':['O1','O0'], 'true':'S0', 'feature':'Yanlis pozitif kimlik alarmi'},
    'S6':  {'obs':['O0','O2'], 'true':'S2', 'feature':'Ilk sensor alarm vermiyor'},
    'S7':  {'obs':['O3','O0'], 'true':'S0', 'feature':'Hatali ag trafik alarmi'},
    'S8':  {'obs':['O1','O2','O3'], 'true':'S3', 'feature':'Cok asamali saldiri'},
    'S9':  {'obs':['O2','O2'], 'true':'S2', 'feature':'Surekli yanal hareket belirtisi'},
    'S10': {'obs':['O1','O3'], 'true':'S3', 'feature':'Kimlik ele gecirme ve veri cikisi'},
    'S11': {'obs':['O0','O1','O0'], 'true':'S1', 'feature':'Dusuk ve belirsiz saldiri'},
    'S12': {'obs':['O1','O2','O3','O3'], 'true':'S3', 'feature':'Tam stres senaryosu'},
}

# Raw LLM decisions are intentionally stored before probabilistic control.
raw_llm = {
    'S1':  {'action':'A0','p_attack':0.08,'reason':'Belirgin anomali yok; izleme yeterli gorunuyor.','errors':['E0']},
    'S2':  {'action':'A3','p_attack':0.90,'reason':'Iki kimlik anomalisi hesabin ele gecirildigini guclu sekilde dusunduruyor; oturumu sonlandirip sifirlamak gerekir.','errors':['E11']},
    'S3':  {'action':'A4','p_attack':0.92,'reason':'Kimlik anomalisi ve uc nokta alarmi birlikte yanal hareket riskini yukseltiyor; cihazi izole etmek gerekir.','errors':['E8']},
    'S4':  {'action':'A4','p_attack':0.98,'reason':'Uc nokta alarmindan sonra veri cikisi kritik veri sizdirma belirtisidir; acil izolasyon gerekir.','errors':['E12']},
    'S5':  {'action':'A3','p_attack':0.75,'reason':'Kimlik anomalisi goruldugu icin hesap guvenligi icin sifirlama yapilmalidir.','errors':['E2','E5','E6']},
    'S6':  {'action':'A1','p_attack':0.55,'reason':'Ilk gozlem normal olsa da sonraki uc nokta alarmi nedeniyle ek inceleme yapilmalidir.','errors':['E3']},
    'S7':  {'action':'A4','p_attack':0.85,'reason':'Olagan disi veri cikisi olasi sizdirma olarak kabul edilip cihaz izole edilmelidir.','errors':['E2','E5','E11']},
    'S8':  {'action':'A4','p_attack':0.99,'reason':'Kimlik, uc nokta ve veri cikisi sinyalleri cok asamali kritik saldiri gosteriyor; izolasyon gerekir.','errors':['E12']},
    'S9':  {'action':'A4','p_attack':0.95,'reason':'Tekrarlayan uc nokta alarmlari yanal hareketi guclu sekilde destekliyor.','errors':['E0']},
    'S10': {'action':'A4','p_attack':0.97,'reason':'Kimlik anomalisi ile veri cikisinin birlikte gorulmesi kritik risktir; izolasyon gerekir.','errors':['E12']},
    'S11': {'action':'A0','p_attack':0.25,'reason':'Sinyaller zayif ve celiskili; simdilik izleme yeterli.','errors':['E6']},
    'S12': {'action':'A4','p_attack':0.995,'reason':'Ardisik kritik sinyaller veri sizdirma riskini cok yuksek yapar; acil izolasyon gerekir.','errors':['E12']},
}


def normalize(d, keys=active_states):
    s = sum(max(0.0, d.get(k,0.0)) for k in keys)
    if s <= 0:
        return {k:1.0/len(keys) for k in keys}
    return {k:max(0.0,d.get(k,0.0))/s for k in keys}


def bayes_update(belief, obs):
    numer = {s: belief.get(s,0.0)*obs_prob[s][obs] for s in active_states}
    return normalize(numer)


def static_nb_belief(obs_seq):
    w = {s:prior[s] for s in active_states}
    for o in obs_seq:
        for s in active_states:
            w[s] *= obs_prob[s][o]
    return normalize(w)


def predict_dynamic(belief):
    out = {s:0.0 for s in active_states}
    # Ignore S4 in observational diagnosis before intervention; renormalize active mass.
    for s in active_states:
        for sp in active_states:
            out[sp] += belief.get(s,0.0)*T[s][sp]
    return normalize(out)


def dynamic_belief_trace(obs_seq):
    b = {s:prior[s] for s in active_states}
    trace=[]
    for i,o in enumerate(obs_seq,1):
        if i>1:
            b = predict_dynamic(b)
        b = bayes_update(b,o)
        trace.append((i,o,b.copy()))
    return trace


def entropy(b):
    return -sum(p*math.log(p+1e-15,2) for p in b.values())


def expected_utility(b, a):
    # Source does not provide a numeric delay coefficient; immediate EU uses delay penalty=0.
    security_gain = 0.0
    for s in ['S1','S2','S3']:
        security_gain += b.get(s,0.0)*harm[s]*contain[a][s]
    cost = action_cost[a]
    wrong_pen = b.get('S0',0.0)*normal_penalty[a]
    delay_pen = 0.0
    eu = security_gain - cost - wrong_pen - delay_pen
    return {
        'security_gain':security_gain,'cost':cost,'wrong_penalty':wrong_pen,
        'delay_penalty':delay_pen,'eu':eu
    }


def best_action_eu(b):
    vals={a:expected_utility(b,a)['eu'] for a in actions}
    # deterministic tie break toward less costly action
    best=max(actions, key=lambda a:(vals[a],-action_cost[a]))
    return best, vals[best], vals


def test_outcome_probs(b,test):
    sens, spec = tests[test]['sens'],tests[test]['spec']
    p_attack=sum(b.get(s,0.0) for s in ['S1','S2','S3'])
    p_normal=b.get('S0',0.0)
    # Active beliefs sum to 1.
    p_pos=sens*p_attack+(1-spec)*p_normal
    return p_pos,1-p_pos


def update_binary_test(b,test,positive):
    sens,spec=tests[test]['sens'],tests[test]['spec']
    w={}
    for s in active_states:
        if s=='S0':
            likelihood=(1-spec) if positive else spec
        else:
            likelihood=sens if positive else (1-sens)
        w[s]=b[s]*likelihood
    return normalize(w)


def voi(b,test):
    a0, base_eu,_=best_action_eu(b)
    ppos,pneg=test_outcome_probs(b,test)
    bpos=update_binary_test(b,test,True)
    bneg=update_binary_test(b,test,False)
    apos,eupos,_=best_action_eu(bpos)
    aneg,euneg,_=best_action_eu(bneg)
    ev_after=ppos*eupos+pneg*euneg
    bd=ev_after-base_eu-tests[test]['cost']
    return {'base_action':a0,'base_eu':base_eu,'p_pos':ppos,'post_pos_action':apos,'post_pos_eu':eupos,
            'post_neg_action':aneg,'post_neg_eu':euneg,'ev_after':ev_after,'cost':tests[test]['cost'],'voi':bd}


def pomdp_action(b):
    p0=b['S0']; p1=b['S1']; p2=b['S2']; p3=b['S3']; patt=1-p0
    # Follow supplied rules with priority for critical states.
    if p3>0.40:
        return 'A4', True, 'P(S3)>0.40: acil izolasyon ve insan yonlendirmesi'
    if p2>0.55:
        return 'A4', False, 'P(S2)>0.55: cihaz izolasyonu degerlendirildi'
    if p1>0.60:
        # choose between MFA and account reset by EU, limited to A2/A3
        eu2=expected_utility(b,'A2')['eu']; eu3=expected_utility(b,'A3')['eu']
        a='A3' if eu3>=eu2 else 'A2'
        return a, False, 'P(S1)>0.60: MFA/hesap mudahalesi'
    if p0>0.80:
        return 'A0',False,'P(S0)>0.80: izle'
    if 0.40 < patt <= 0.65:
        return 'A1',False,'0.40 < saldiri olasiligi <= 0.65: ek inceleme'
    # uncertain/high but no state-specific threshold: choose EU, but avoid heavy A4 if p0 high.
    a,_,_=best_action_eu(b)
    return a, a=='A5', 'Belirsiz belief: beklenen fayda ile secim'


def bdk_action_before_test(b):
    # Critical risk is handled before VOI, as instructed.
    a,human,reason=pomdp_action(b)
    if b['S3']>0.40 or b['S2']>0.70:
        return {'test':None,'action':a,'human':human,'reason':'Kritik risk: BD beklenmeden mudahale. '+reason}
    # High uncertainty or moderate attack risk -> calculate all VOIs.
    voivals={t:voi(b,t) for t in tests}
    best_t=max(tests,key=lambda t:voivals[t]['voi'])
    if voivals[best_t]['voi']>0 and (entropy(b)>1.15 or (0.25 < 1-b['S0'] < 0.75)):
        return {'test':best_t,'action':None,'human':False,'reason':f"Pozitif Bilgi Degeri: {best_t} secildi",'voi':voivals[best_t]['voi']}
    return {'test':None,'action':a,'human':human,'reason':'Pozitif BD bulunmadi veya belirsizlik dusuk. '+reason}


def sample_test(test,true_state,rng):
    t=tests[test]
    if true_state=='S0':
        ppos=1-t['spec']
    else:
        ppos=t['sens']
    return rng.random()<ppos


def action_transition(s,a):
    if s=='S4':
        # action cannot improve beyond contained; apply base transition to keep dynamics realistic.
        return T['S4'].copy()
    if s=='S0':
        return T['S0'].copy()
    ps=contain[a][s]
    dist={sp:(1-ps)*T[s][sp] for sp in states}
    dist['S4'] += ps
    # normalize numerical residue
    total=sum(dist.values())
    return {sp:v/total for sp,v in dist.items()}


def mdp_reward(s,a):
    # Per-step risk-sensitive cost: current damage + action cost + unnecessary normal penalty.
    risk_cost=max(harm[s],0)
    return -(risk_cost + action_cost[a] + (normal_penalty[a] if s=='S0' else 0))


def solve_mdp():
    V={s:0.0 for s in states}
    for _ in range(10000):
        new={}
        for s in states:
            qs=[]
            for a in actions:
                dist=action_transition(s,a)
                q=mdp_reward(s,a)+GAMMA*sum(dist[sp]*V[sp] for sp in states)
                qs.append((q,a))
            new[s]=max(qs)[0]
        if max(abs(new[s]-V[s]) for s in states)<1e-10:
            V=new; break
        V=new
    policy={}; qbest={}
    for s in states:
        qs={}
        for a in actions:
            dist=action_transition(s,a)
            qs[a]=mdp_reward(s,a)+GAMMA*sum(dist[sp]*V[sp] for sp in states)
        a=max(actions,key=lambda x:qs[x])
        policy[s]=a; qbest[s]=qs[a]
    return policy,qbest,V

mdp_policy,mdp_q,mdp_V=solve_mdp()

# Single O1 posterior table
numer={s:prior[s]*obs_prob[s]['O1'] for s in active_states}
den=sum(numer.values())
o1_post={s:numer[s]/den for s in active_states}

# Belief traces and scenario-level analysis
belief_rows=[]
scenario_beliefs={}
nb_beliefs={}
for sc,info in scenarios.items():
    tr=dynamic_belief_trace(info['obs'])
    for t,o,b in tr:
        belief_rows.append([sc,t,o,b['S0'],b['S1'],b['S2'],b['S3'],max(active_states,key=lambda s:b[s])])
    scenario_beliefs[sc]=tr[-1][2]
    nb_beliefs[sc]=static_nb_belief(info['obs'])

# EU and VOI tables
utility_rows=[]
voi_rows=[]
for sc,b in scenario_beliefs.items():
    for a in actions:
        x=expected_utility(b,a)
        utility_rows.append([sc,a,action_name[a],x['security_gain'],x['cost'],x['wrong_penalty'],x['delay_penalty'],x['eu']])
    for t in tests:
        x=voi(b,t)
        voi_rows.append([sc,t,tests[t]['name'],x['base_eu'],x['ev_after'],x['cost'],x['voi'],'Test yapilabilir' if x['voi']>0 else 'Test yapilmamali'])

# MDP table
mdp_rows=[]
for s in states:
    a=mdp_policy[s]
    one_b={x:0.0 for x in active_states}
    if s in active_states: one_b[s]=1.0
    eu=expected_utility(one_b,a)['eu'] if s in active_states else -action_cost[a]
    mdp_rows.append([s,state_name[s],a,action_name[a],eu,mdp_q[s],'Tam gozlemlenebilir MDP value iteration, gamma=0.90'])

# LLM table
llm_rows=[]
for sc,x in raw_llm.items():
    has_prob='Evet' if x.get('p_attack') is not None else 'Hayir'
    wrong_cert='Evet' if 'E11' in x['errors'] else 'Hayir'
    cost_considered='Hayir' if any(e in x['errors'] for e in ['E2','E7','E8']) else 'Kismen'
    llm_rows.append([sc,x['action'],action_name[x['action']],x['p_attack'],x['reason'],has_prob,wrong_cert,cost_considered,','.join(x['errors'])])

# Model decision helpers

def m0_distribution(obs_seq):
    weights={'O0':0,'O1':1,'O2':2,'O3':3}
    score=sum(weights[o] for o in obs_seq)
    # fixed-threshold calibration proxy (explicit experimental assumption)
    p_attack={0:0.08,1:0.25,2:0.45,3:0.65,4:0.80,5:0.90}.get(score,0.96)
    # distribute attack mass by strongest observation
    if 'O3' in obs_seq:
        mix={'S1':0.10,'S2':0.20,'S3':0.70}
    elif 'O2' in obs_seq:
        mix={'S1':0.15,'S2':0.70,'S3':0.15}
    else:
        mix={'S1':0.75,'S2':0.20,'S3':0.05}
    b={'S0':1-p_attack}
    for s in ['S1','S2','S3']:
        b[s]=p_attack*mix[s]
    return normalize(b)


def m0_action(obs_seq):
    weights={'O0':0,'O1':1,'O2':2,'O3':3}
    score=sum(weights[o] for o in obs_seq)
    # Fixed threshold >=3 triggers intervention; escalation by strongest signal.
    if score<3: return 'A0',False
    if 'O3' in obs_seq: return 'A4',False
    if 'O2' in obs_seq: return 'A4',False
    return 'A3',False


def llm_distribution(sc):
    p=raw_llm[sc]['p_attack']
    obs=scenarios[sc]['obs']
    # Overconfident but plausible allocation to a dominant attack class.
    if 'O3' in obs:
        dom='S3'
    elif 'O2' in obs:
        dom='S2'
    else:
        dom='S1'
    b={'S0':1-p,'S1':0.0,'S2':0.0,'S3':0.0}
    b[dom]=p*0.85
    remaining=p-b[dom]
    for s in ['S1','S2','S3']:
        if s!=dom: b[s]=remaining/2
    return normalize(b)


def choose_model(model,sc,rng):
    info=scenarios[sc]; true=info['true']; obs=info['obs']
    test=None; test_positive=None; human=False
    if model=='M0':
        b=m0_distribution(obs); action,human=m0_action(obs); reason='Sabit alarm esigi'
    elif model=='M1':
        b=nb_beliefs[sc].copy(); action,_,_=best_action_eu(b); reason='Naive Bayes + tek adimli EU'
    elif model=='M2':
        b=scenario_beliefs[sc].copy(); action,_,_=best_action_eu(b); reason='Dinamik Bayes belief + tek adimli EU'
    elif model=='M3':
        b=scenario_beliefs[sc].copy()
        v={t:voi(b,t) for t in tests}; best_t=max(tests,key=lambda t:v[t]['voi'])
        if v[best_t]['voi']>0:
            test=best_t; test_positive=sample_test(test,true,rng); b=update_binary_test(b,test,test_positive)
        action,_,_=best_action_eu(b); reason='Karar agi + Bilgi Degeri'
    elif model=='M4':
        b={s:(1.0 if s==true else 0.0) for s in active_states}
        action=mdp_policy[true]; reason='Oracle MDP (gercek durum biliniyor)'
    elif model=='M5':
        b=scenario_beliefs[sc].copy(); action,human,reason=pomdp_action(b)
    elif model=='M6':
        b=scenario_beliefs[sc].copy(); d=bdk_action_before_test(b)
        if d.get('test'):
            test=d['test']; test_positive=sample_test(test,true,rng); b=update_binary_test(b,test,test_positive)
            action,human,reason=pomdp_action(b)
            reason='BDK-POMDP: '+d['reason']+'; test sonrasi '+reason
        else:
            action=d['action']; human=d['human']; reason='BDK-POMDP: '+d['reason']
    elif model=='M7':
        b=llm_distribution(sc); action=raw_llm[sc]['action']; reason='Ham LLM: '+raw_llm[sc]['reason']
    elif model=='M8':
        # Controller uses BDK-POMDP belief and can override raw LLM action.
        b=scenario_beliefs[sc].copy(); d=bdk_action_before_test(b)
        if d.get('test'):
            test=d['test']; test_positive=sample_test(test,true,rng); b=update_binary_test(b,test,test_positive)
            ctrl_action,human,ctrl_reason=pomdp_action(b)
        else:
            ctrl_action=d['action']; human=d['human']; ctrl_reason=d['reason']
        llm_action=raw_llm[sc]['action']
        # Accept raw LLM only if it matches controller or is adjacent low-risk substitute with close EU.
        if llm_action==ctrl_action:
            action=llm_action; override=False
        else:
            eu_llm=expected_utility(b,llm_action)['eu']; eu_ctrl=expected_utility(b,ctrl_action)['eu']
            if abs(eu_ctrl-eu_llm)<=2 and action_cost[llm_action]<=action_cost[ctrl_action]+5:
                action=llm_action; override=False
            else:
                action=ctrl_action; override=True
        reason=f"LLM + BDK-POMDP; raw={llm_action}, kontrol={ctrl_action}, override={override}. {ctrl_reason}"
    else:
        raise ValueError(model)
    return b,action,human,test,test_positive,reason

# Correct decision reference = MDP action for the true state. Also accept A5 for active attacks if MDP action is A4/A3 and human referral is selected.
def is_correct_decision(true_state,action,human=False):
    if true_state=='S0':
        return action in ['A0','A1']
    ref=mdp_policy[true_state]
    if action==ref: return True
    # Risk-equivalent acceptable alternatives, grounded in supplied action success table.
    if true_state=='S1' and action in ['A2','A3','A5']: return True
    if true_state=='S2' and action in ['A4','A5']: return True
    if true_state=='S3' and action in ['A4','A5']: return True
    return False


def brier(b,true_state):
    return sum((b.get(s,0.0)-(1.0 if s==true_state else 0.0))**2 for s in active_states)/len(active_states)

# 540 experiment runs
models=[f'M{i}' for i in range(9)]
run_rows=[]
run_records=[]
for rep in range(1,REPEATS+1):
    for sc in scenarios:
        for model in models:
            rng=random.Random(SEED*100000 + rep*1000 + int(sc[1:])*10 + int(model[1:]))
            b,action,human,test,test_pos,reason=choose_model(model,sc,rng)
            true=scenarios[sc]['true']
            attacked=true in ['S1','S2','S3']
            pcont=contain[action][true] if attacked else 0.0
            # Common random number across models for fair containment comparison.
            u_cont=random.Random(SEED*300000 + rep*1000 + int(sc[1:])).random()
            contained=(u_cont<pcont) if attacked else False
            false_iso=(true=='S0' and action=='A4')
            delay=len(scenarios[sc]['obs']) + (1 if test else 0)
            intervened=action!='A0'
            correct=is_correct_decision(true,action,human)
            rec={
                'repeat':rep,'scenario':sc,'model':model,'true_state':true,
                'observations':','.join(scenarios[sc]['obs']),
                'p_s0':b['S0'],'p_s1':b['S1'],'p_s2':b['S2'],'p_s3':b['S3'],
                'action':action,'action_name':action_name[action],'human_referral':human,
                'test':test or '','test_positive':'' if test is None else int(bool(test_pos)),
                'contained':contained,'false_isolation':false_iso,'correct_decision':correct,
                'delay_steps':delay,'intervened':intervened,'brier':brier(b,true),
                'reason':reason
            }
            run_records.append(rec)
            run_rows.append(list(rec.values()))

# Metrics by model
metric_rows=[]
metric_map={}
for model in models:
    rows=[r for r in run_records if r['model']==model]
    attack=[r for r in rows if r['true_state'] in ['S1','S2','S3']]
    normal=[r for r in rows if r['true_state']=='S0']
    tests_done=[r for r in rows if r['test']]
    # Helpful test if final action differs from the action that would be selected before the test.
    helpful=0
    for r in tests_done:
        sc=r['scenario']; base_b=scenario_beliefs[sc]
        if model=='M3': base_a=best_action_eu(base_b)[0]
        elif model in ['M6','M8']: base_a=pomdp_action(base_b)[0]
        else: base_a=r['action']
        if r['action']!=base_a: helpful += 1
    oso=100*sum(r['contained'] for r in attack)/len(attack)
    yio=100*sum(r['false_isolation'] for r in normal)/len(normal)
    dko=100*sum(r['correct_decision'] for r in rows)/len(rows)
    iyo=100*sum(r['human_referral'] for r in rows)/len(rows)
    int_rows=[r for r in rows if r['intervened']]
    omg=sum(r['delay_steps'] for r in int_rows)/len(int_rows) if int_rows else 0.0
    bto=100*len(tests_done)/len(rows)
    ybto=100*helpful/len(tests_done) if tests_done else 0.0
    bs=sum(r['brier'] for r in rows)/len(rows)
    cal=max(0,100-bs*100)
    delay_success=max(0,100-(omg/MAX_ALLOWED_STEP)*100)
    mgs=0.30*oso+0.20*(100-yio)+0.20*dko+0.15*cal+0.10*delay_success+0.05*ybto
    level='Cok guvenilir' if mgs>=85 else 'Guvenilir' if mgs>=70 else 'Orta' if mgs>=50 else 'Zayif' if mgs>=30 else 'Kritik'
    vals={'OSO':oso,'YIO':yio,'DKO':dko,'IYO':iyo,'OMG':omg,'BTO':bto,'YBTO':ybto,'Brier':bs,'Kalibrasyon':cal,'GecikmeBasarisi':delay_success,'MGS':mgs,'Duzey':level}
    metric_map[model]=vals
    metric_rows.append([model,oso,yio,dko,iyo,omg,bto,ybto,bs,cal,delay_success,mgs,level])

# Ablation: map components to closest model/controlled variants; simulate explicitly via combinations.
ablation_defs={
    'AB0':'Yalnizca sabit alarm esigi',
    'AB1':'Bayes guncellemesi var, Bilgi Degeri yok',
    'AB2':'POMDP var, test maliyeti yok',
    'AB3':'POMDP + test maliyeti',
    'AB4':'POMDP + Bilgi Degeri',
    'AB5':'POMDP + Bilgi Degeri + insan yonlendirme kapisi',
}

def choose_ablation(ab,sc,rng):
    true=scenarios[sc]['true']; b=scenario_beliefs[sc].copy(); test=None; test_pos=None; human=False
    if ab=='AB0':
        b=m0_distribution(scenarios[sc]['obs']); action,human=m0_action(scenarios[sc]['obs'])
    elif ab=='AB1':
        action,_,_=best_action_eu(b)
    elif ab=='AB2':
        # POMDP with free testing: pick test maximizing expected post-test EU without cost, whenever uncertain.
        candidates={}
        for t in tests:
            x=voi(b,t); candidates[t]=x['ev_after']-x['base_eu']  # cost removed
        best=max(candidates,key=candidates.get)
        if candidates[best]>0 and entropy(b)>0.8:
            test=best; test_pos=sample_test(test,true,rng); b=update_binary_test(b,test,test_pos)
        action,human,_=pomdp_action(b)
    elif ab=='AB3':
        # POMDP with cost-aware tests but no explicit positive-VOI gate: test if gross gain exceeds cost/2.
        v={t:voi(b,t) for t in tests}; best=max(v,key=lambda t:v[t]['voi'])
        if (v[best]['ev_after']-v[best]['base_eu']) > tests[best]['cost']/2 and entropy(b)>0.8:
            test=best; test_pos=sample_test(test,true,rng); b=update_binary_test(b,test,test_pos)
        action,human,_=pomdp_action(b)
    elif ab=='AB4':
        d=bdk_action_before_test(b)
        if d.get('test'):
            test=d['test']; test_pos=sample_test(test,true,rng); b=update_binary_test(b,test,test_pos)
        action,human,_=pomdp_action(b)
        human=False  # no human gate in AB4
    elif ab=='AB5':
        d=bdk_action_before_test(b)
        if d.get('test'):
            test=d['test']; test_pos=sample_test(test,true,rng); b=update_binary_test(b,test,test_pos)
        action,human,_=pomdp_action(b)
    return b,action,human,test,test_pos

ab_rows=[]; ab_run_records=[]
for ab in ablation_defs:
    for rep in range(1,REPEATS+1):
        for sc in scenarios:
            rng=random.Random(SEED*200000+rep*1000+int(sc[1:])*10+int(ab[-1]))
            b,a,h,t,tp=choose_ablation(ab,sc,rng); true=scenarios[sc]['true']
            attacked=true in ['S1','S2','S3']; u_cont=random.Random(SEED*400000+rep*1000+int(sc[1:])).random(); contained=(u_cont<contain[a][true]) if attacked else False
            rec={'ablation':ab,'repeat':rep,'scenario':sc,'true_state':true,'action':a,'human':h,'test':t or '',
                 'contained':contained,'false_iso':true=='S0' and a=='A4','correct':is_correct_decision(true,a,h),
                 'delay':len(scenarios[sc]['obs'])+(1 if t else 0),'intervened':a!='A0','brier':brier(b,true),
                 'test_cost':tests[t]['cost'] if t else 0}
            ab_run_records.append(rec)

for ab in ablation_defs:
    rr=[r for r in ab_run_records if r['ablation']==ab]; atk=[r for r in rr if r['true_state'] in ['S1','S2','S3']]; norm=[r for r in rr if r['true_state']=='S0']
    oso=100*sum(r['contained'] for r in atk)/len(atk); yio=100*sum(r['false_iso'] for r in norm)/len(norm); dko=100*sum(r['correct'] for r in rr)/len(rr)
    tc=sum(r['test_cost'] for r in rr); ints=[r for r in rr if r['intervened']]; omg=sum(r['delay'] for r in ints)/len(ints) if ints else 0
    bs=sum(r['brier'] for r in rr)/len(rr); cal=max(0,100-bs*100); ds=max(0,100-(omg/MAX_ALLOWED_STEP)*100)
    tests_done=[r for r in rr if r['test']]; ybto=0.0
    # MGS with YBTO=0 for ablation summary to isolate structure and because "karari degistiren" detail is not needed in Table 21.
    mgs=0.30*oso+0.20*(100-yio)+0.20*dko+0.15*cal+0.10*ds+0.05*ybto
    ab_rows.append([ab,ablation_defs[ab],oso,yio,dko,tc,omg,mgs])

# POMDP scenario summary (single deterministic decision with most likely/no sampled test: M5)
pomdp_rows=[]
for sc in scenarios:
    b=scenario_beliefs[sc]; a,h,reason=pomdp_action(b); true=scenarios[sc]['true']
    pomdp_rows.append([sc,b['S0'],b['S1'],b['S2'],b['S3'],a,action_name[a],true,'Evet' if is_correct_decision(true,a,h) else 'Hayir',reason])

# BDK-POMDP deterministic plan summary before actual test result
bdk_rows=[]
for sc,b in scenario_beliefs.items():
    d=bdk_action_before_test(b)
    bdk_rows.append([sc,b['S0'],b['S1'],b['S2'],b['S3'],d.get('test') or '',d.get('action') or '',d.get('reason','')])

# Output helpers

def write_csv(filename,headers,rows):
    with open(OUT/filename,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(headers); w.writerows(rows)

write_csv('prior_check.csv',['Durum','Olasilik','Toplama_Katkisi'],[[s,prior[s],prior[s]] for s in active_states]+[['Toplam',sum(prior[s] for s in active_states),1.0]])
write_csv('bayes_o1.csv',['Durum','Onsel','P(O1|S)','Carpim','Sonsal'],[[s,prior[s],obs_prob[s]['O1'],numer[s],o1_post[s]] for s in active_states]+[['Toplam','','',den,1.0]])
write_csv('belief_updates.csv',['Senaryo','Zaman','Gozlem','P(S0)','P(S1)','P(S2)','P(S3)','En_Olasi_Durum'],belief_rows)
write_csv('expected_utility.csv',['Senaryo','Eylem','Eylem_Adi','Beklenen_Guvenlik_Kazanci','Mudahale_Maliyeti','Yanlis_Mudahale_Cezasi','Gecikme_Cezasi','Beklenen_Fayda'],utility_rows)
write_csv('value_of_information.csv',['Senaryo','Test','Test_Adi','Testsiz_En_Iyi_Fayda','Test_Sonrasi_Beklenen_Fayda','Test_Maliyeti','Bilgi_Degeri','Karar'],voi_rows)
write_csv('mdp_policy.csv',['Durum','Durum_Adi','Eylem','Eylem_Adi','Tek_Adim_EU','MDP_Q','Gerekce'],mdp_rows)
write_csv('pomdp_results.csv',['Senaryo','P(S0)','P(S1)','P(S2)','P(S3)','Secilen_Eylem','Eylem_Adi','Gercek_Durum','Dogru_Karar','Gerekce'],pomdp_rows)
write_csv('bdk_pomdp_results.csv',['Senaryo','P(S0)','P(S1)','P(S2)','P(S3)','Secilen_Test','Dogrudan_Eylem','Gerekce'],bdk_rows)
write_csv('raw_llm_decisions.csv',['Senaryo','Eylem','Eylem_Adi','LLM_Saldiri_Olasiligi','Gerekce','Olasilik_Verdi','Yanlis_Kesinlik','Fayda_Maliyet','Hata_Kodlari'],llm_rows)
write_csv('experiment_540_runs.csv',list(run_records[0].keys()),[list(r.values()) for r in run_records])
write_csv('general_metrics.csv',['Model','OSO','YIO','DKO','IYO','OMG','BTO','YBTO','Brier','Kalibrasyon_Basarisi','Gecikme_Basarisi','MGS','Duzey'],metric_rows)
write_csv('ablation_results.csv',['Yapilandirma','Aciklama','OSO','YIO','DKO','Test_Maliyeti','OMG','MGS'],ab_rows)

# Evaluation answers, grounded in generated results.
best_mgs=max(metric_map,key=lambda m:metric_map[m]['MGS'])
best_deploy=max([m for m in models if m!='M4'],key=lambda m:metric_map[m]['MGS'])
highest_yio=max(metric_map,key=lambda m:metric_map[m]['YIO'])
highest_oso=max(metric_map,key=lambda m:metric_map[m]['OSO'])
# Positive/negative VOI scenarios/tests based on deterministic analysis
positive_voi=[(sc,t,x) for sc in scenarios for t in tests for x in [voi(scenario_beliefs[sc],t)['voi']] if x>0]
negative_voi=[(sc,t,x) for sc in scenarios for t in tests for x in [voi(scenario_beliefs[sc],t)['voi']] if x<0]

summary={
    'seed':SEED,'repeats':REPEATS,'gamma':GAMMA,'max_allowed_step':MAX_ALLOWED_STEP,
    'o1_posterior':o1_post,'mdp_policy':mdp_policy,
    'best_mgs_model':best_mgs,'best_deployable_model':best_deploy,
    'highest_yio_model':highest_yio,'highest_oso_model':highest_oso,
    'positive_voi_count':len(positive_voi),'negative_voi_count':len(negative_voi),
    'metrics':metric_map,
    'assumptions':[
        'Dokumanda gecikme cezasi icin sayisal katsayi verilmediginden tek-adimli EU hesaplarinda gecikme cezasi 0 alindi; gecikme MDP/POMDP gecisleri ve OMG ile olculdu.',
        'Bilgi testleri ikili saldiri/normal testi olarak modellendi; duyarlilik S1-S3 icin, ozgulluk S0 icin uygulandi.',
        'M2 Bayes Agi icin ayri CPT verilmediginden Tablo 7 gecisleriyle dinamik Bayes agi (S_t -> S_t+1 -> O_t) kullanildi.',
        'MGS gecikme basarisi icin maksimum izinli adim dokumanda sayisal verilmediginden 5 adim secildi (en uzun 4 gozlem + 1 ek test).',
        'M0 sabit esik icin dokumanda alarm puan fonksiyonu verilmediginden O0=0, O1=1, O2=2, O3=3 ve esik=3 deneysel varsayimi kullanildi.',
        'Ham LLM olasilik dagilimi, kaydedilen saldiri olasiligini gozlemle en uyumlu baskin saldiri sinifina dagitan tekrarlanabilir bir kalibrasyon temsiliyle Brier hesabina cevrildi.'
    ]
}
with open(OUT/'summary.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)

print(json.dumps({
    'O1 posterior':o1_post,
    'MDP policy':mdp_policy,
    'Best MGS':(best_mgs,metric_map[best_mgs]['MGS']),
    'Best deployable':(best_deploy,metric_map[best_deploy]['MGS']),
    'Highest YIO':(highest_yio,metric_map[highest_yio]['YIO']),
    'Highest OSO':(highest_oso,metric_map[highest_oso]['OSO']),
    'Positive VOI count':len(positive_voi),
    'Metrics':metric_map,
    'Ablation':ab_rows,
},ensure_ascii=False,indent=2))
