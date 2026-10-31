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
    'S0':'Normal','S1':'Credential Compromise','S2':'Lateral Movement','S3':'Data Exfiltration','S4':'Contained'
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
    'A0':'Continue Monitoring',
    'A1':'Perform Additional Log and EDR Review',
    'A2':'Request MFA Verification',
    'A3':'Terminate Suspicious Session and Reset Account',
    'A4':'Isolate Endpoint',
    'A5':'Escalate to Human Security Expert',
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
    'T1': {'name':'Detailed Authentication Log','cost':3,'sens':0.78,'spec':0.82},
    'T2': {'name':'MFA Verification Test','cost':5,'sens':0.85,'spec':0.90},
    'T3': {'name':'Deep EDR Scan','cost':8,'sens':0.90,'spec':0.88},
    'T4': {'name':'Network Traffic Analysis','cost':7,'sens':0.88,'spec':0.85},
    'T5': {'name':'Human Expert Verification','cost':12,'sens':0.95,'spec':0.93},
}

scenarios = {
    'S1':  {'obs':['O0','O0'], 'true':'S0', 'feature':'Normal User Behavior'},
    'S2':  {'obs':['O1','O1'], 'true':'S1', 'feature':'Repeated Authentication Anomaly'},
    'S3':  {'obs':['O1','O2'], 'true':'S2', 'feature':'Endpoint Alert Following Authentication Anomaly'},
    'S4':  {'obs':['O2','O3'], 'true':'S3', 'feature':'Data Exfiltration Following Lateral Movement'},
    'S5':  {'obs':['O1','O0'], 'true':'S0', 'feature':'False-Positive Authentication Alert'},
    'S6':  {'obs':['O0','O2'], 'true':'S2', 'feature':'First Sensor Does Not Trigger an Alert'},
    'S7':  {'obs':['O3','O0'], 'true':'S0', 'feature':'Incorrect Network Traffic Alert'},
    'S8':  {'obs':['O1','O2','O3'], 'true':'S3', 'feature':'Multi-Stage Attack'},
    'S9':  {'obs':['O2','O2'], 'true':'S2', 'feature':'Persistent Signs of Lateral Movement'},
    'S10': {'obs':['O1','O3'], 'true':'S3', 'feature':'Credential Compromise and Data Exfiltration'},
    'S11': {'obs':['O0','O1','O0'], 'true':'S1', 'feature':'Low-Level and Uncertain Attack'},
    'S12': {'obs':['O1','O2','O3','O3'], 'true':'S3', 'feature':'Full Stress-Test Scenario'},
}

# Raw LLM decisions are intentionally stored before probabilistic control.
raw_llm = {
    'S1':  {'action':'A0','p_attack':0.08,'reason':'No significant anomaly is present; monitoring appears sufficient.','errors':['E0']},
    'S2':  {'action':'A3','p_attack':0.90,'reason':'Two authentication anomalies strongly suggest account compromise; the session should be terminated and the account reset.','errors':['E11']},
    'S3':  {'action':'A4','p_attack':0.92,'reason':'The authentication anomaly and endpoint alert together increase the risk of lateral movement; the device should be isolated.','errors':['E8']},
    'S4':  {'action':'A4','p_attack':0.98,'reason':'Data exfiltration following an endpoint alert is a critical sign of data leakage; immediate isolation is required.','errors':['E12']},
    'S5':  {'action':'A3','p_attack':0.75,'reason':'Because an authentication anomaly was detected, the account should be reset to protect its security.','errors':['E2','E5','E6']},
    'S6':  {'action':'A1','p_attack':0.55,'reason':'Although the first observation is normal, further investigation is required because of the subsequent endpoint alert.','errors':['E3']},
    'S7':  {'action':'A4','p_attack':0.85,'reason':'Unusual data exfiltration should be treated as a possible leak, and the device should be isolated.','errors':['E2','E5','E11']},
    'S8':  {'action':'A4','p_attack':0.99,'reason':'Authentication, endpoint, and data-exfiltration signals indicate a critical multi-stage attack; isolation is required.','errors':['E12']},
    'S9':  {'action':'A4','p_attack':0.95,'reason':'Repeated endpoint alerts strongly support the presence of lateral movement.','errors':['E0']},
    'S10': {'action':'A4','p_attack':0.97,'reason':'The combination of an authentication anomaly and data exfiltration represents a critical risk; isolation is required.','errors':['E12']},
    'S11': {'action':'A0','p_attack':0.25,'reason':'The signals are weak and conflicting; monitoring is sufficient for now.','errors':['E6']},
    'S12': {'action':'A4','p_attack':0.995,'reason':'Consecutive critical signals make the risk of data exfiltration very high; immediate isolation is required.','errors':['E12']},
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
    # Ignore S4 in observational diagnosis before intervention; renormalize active-state probability mass.
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
    # The source does not provide a numeric delay coefficient; immediate expected utility uses a delay penalty of 0.
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
    # Active-state beliefs sum to 1.
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
    # Follow the supplied rules, prioritizing critical states.
    if p3>0.40:
        return 'A4', True, 'P(S3)>0.40: immediate isolation and human escalation'
    if p2>0.55:
        return 'A4', False, 'P(S2)>0.55: device isolation considered'
    if p1>0.60:
        # choose between MFA and account reset by EU, limited to A2/A3
        eu2=expected_utility(b,'A2')['eu']; eu3=expected_utility(b,'A3')['eu']
        a='A3' if eu3>=eu2 else 'A2'
        return a, False, 'P(S1)>0.60: MFA/account intervention'
    if p0>0.80:
        return 'A0',False,'P(S0)>0.80: monitor'
    if 0.40 < patt <= 0.65:
        return 'A1',False,'0.40 < attack probability <= 0.65: additional investigation'
    # uncertain/high but no state-specific threshold: choose EU, but avoid heavy A4 if p0 high.
    a,_,_=best_action_eu(b)
    return a, a=='A5', 'Uncertain belief: selection based on expected utility'


def bdk_action_before_test(b):
    # Handle critical risk before calculating VOI, as instructed.
    a,human,reason=pomdp_action(b)
    if b['S3']>0.40 or b['S2']>0.70:
        return {'test':None,'action':a,'human':human,'reason':'Critical risk: intervene before VOI. '+reason}
    # High uncertainty or moderate attack risk -> calculate all VOIs.
    voivals={t:voi(b,t) for t in tests}
    best_t=max(tests,key=lambda t:voivals[t]['voi'])
    if voivals[best_t]['voi']>0 and (entropy(b)>1.15 or (0.25 < 1-b['S0'] < 0.75)):
        return {'test':best_t,'action':None,'human':False,'reason':f"Positive Value of Information: {best_t} selected",'voi':voivals[best_t]['voi']}
    return {'test':None,'action':a,'human':human,'reason':'No positive VOI was found or uncertainty is low. '+reason}


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
        voi_rows.append([sc,t,tests[t]['name'],x['base_eu'],x['ev_after'],x['cost'],x['voi'],'Test can be performed' if x['voi']>0 else 'Test should not be performed'])

# MDP table
mdp_rows=[]
for s in states:
    a=mdp_policy[s]
    one_b={x:0.0 for x in active_states}
    if s in active_states: one_b[s]=1.0
    eu=expected_utility(one_b,a)['eu'] if s in active_states else -action_cost[a]
    mdp_rows.append([s,state_name[s],a,action_name[a],eu,mdp_q[s],'Fully Observable MDP Value Iteration, gamma=0.90'])

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
    # Fixed-threshold calibration proxy (explicit experimental assumption)
    p_attack={0:0.08,1:0.25,2:0.45,3:0.65,4:0.80,5:0.90}.get(score,0.96)
    # Distribute attack probability across attack states according to the strongest observation.
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
    # A fixed threshold of >=3 triggers intervention; escalation is based on the strongest signal.
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
        b=m0_distribution(obs); action,human=m0_action(obs); reason='Fixed Alert Threshold'
    elif model=='M1':
        b=nb_beliefs[sc].copy(); action,_,_=best_action_eu(b); reason='Naive Bayes + One-Step Expected Utility'
    elif model=='M2':
        b=scenario_beliefs[sc].copy(); action,_,_=best_action_eu(b); reason='Dynamic Bayesian Belief + One-Step Expected Utility'
    elif model=='M3':
        b=scenario_beliefs[sc].copy()
        v={t:voi(b,t) for t in tests}; best_t=max(tests,key=lambda t:v[t]['voi'])
        if v[best_t]['voi']>0:
            test=best_t; test_positive=sample_test(test,true,rng); b=update_binary_test(b,test,test_positive)
        action,_,_=best_action_eu(b); reason='Decision Tree + Value of Information'
    elif model=='M4':
        b={s:(1.0 if s==true else 0.0) for s in active_states}
        action=mdp_policy[true]; reason='Oracle MDP (true state is known)'
    elif model=='M5':
        b=scenario_beliefs[sc].copy(); action,human,reason=pomdp_action(b)
    elif model=='M6':
        b=scenario_beliefs[sc].copy(); d=bdk_action_before_test(b)
        if d.get('test'):
            test=d['test']; test_positive=sample_test(test,true,rng); b=update_binary_test(b,test,test_positive)
            action,human,reason=pomdp_action(b)
            reason='VOI-POMDP: '+d['reason']+'; after test: '+reason
        else:
            action=d['action']; human=d['human']; reason='VOI-POMDP: '+d['reason']
    elif model=='M7':
        b=llm_distribution(sc); action=raw_llm[sc]['action']; reason='Raw LLM: '+raw_llm[sc]['reason']
    elif model=='M8':
        # The controller uses the VOI-POMDP belief and can override the raw LLM action.
        b=scenario_beliefs[sc].copy(); d=bdk_action_before_test(b)
        if d.get('test'):
            test=d['test']; test_positive=sample_test(test,true,rng); b=update_binary_test(b,test,test_positive)
            ctrl_action,human,ctrl_reason=pomdp_action(b)
        else:
            ctrl_action=d['action']; human=d['human']; ctrl_reason=d['reason']
        llm_action=raw_llm[sc]['action']
        # Accept the raw LLM action only if it matches the controller or is a neighboring low-risk alternative with similar expected utility.
        if llm_action==ctrl_action:
            action=llm_action; override=False
        else:
            eu_llm=expected_utility(b,llm_action)['eu']; eu_ctrl=expected_utility(b,ctrl_action)['eu']
            if abs(eu_ctrl-eu_llm)<=2 and action_cost[llm_action]<=action_cost[ctrl_action]+5:
                action=llm_action; override=False
            else:
                action=ctrl_action; override=True
        reason=f"LLM + VOI-POMDP; raw={llm_action}, controller={ctrl_action}, override={override}. {ctrl_reason}"
    else:
        raise ValueError(model)
    return b,action,human,test,test_positive,reason

# Correct-decision reference = MDP action for the true state. Also accept A5 for active attacks if the MDP action is A4/A3 and human referral is selected.
def is_correct_decision(true_state,action,human=False):
    if true_state=='S0':
        return action in ['A0','A1']
    ref=mdp_policy[true_state]
    if action==ref: return True
    # Risk-equivalent acceptable alternatives, based on the supplied action success table.
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
            # Use a common random number across models for a fair containment comparison.
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
    # A test is helpful if the final action differs from the action that would have been selected before testing.
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
    level='Very Reliable' if mgs>=85 else 'Reliable' if mgs>=70 else 'Moderate' if mgs>=50 else 'Weak' if mgs>=30 else 'Critical'
    vals={'OSO':oso,'YIO':yio,'DKO':dko,'IYO':iyo,'OMG':omg,'BTO':bto,'YBTO':ybto,'Brier':bs,'Kalibrasyon':cal,'GecikmeBasarisi':delay_success,'MGS':mgs,'Level':level}
    metric_map[model]=vals
    metric_rows.append([model,oso,yio,dko,iyo,omg,bto,ybto,bs,cal,delay_success,mgs,level])

# Ablation: map components to the closest model/controlled variants; simulate explicitly through combinations.
ablation_defs={
    'AB0':'Fixed alert threshold only',
    'AB1':'Bayesian update enabled, no Value of Information',
    'AB2':'POMDP enabled, no test cost',
    'AB3':'POMDP + test cost',
    'AB4':'POMDP + Value of Information',
    'AB5':'POMDP + Value of Information + human escalation gate',
}

def choose_ablation(ab,sc,rng):
    true=scenarios[sc]['true']; b=scenario_beliefs[sc].copy(); test=None; test_pos=None; human=False
    if ab=='AB0':
        b=m0_distribution(scenarios[sc]['obs']); action,human=m0_action(scenarios[sc]['obs'])
    elif ab=='AB1':
        action,_,_=best_action_eu(b)
    elif ab=='AB2':
        # POMDP with free testing: select the test that maximizes expected post-test utility without cost whenever uncertainty is high.
        candidates={}
        for t in tests:
            x=voi(b,t); candidates[t]=x['ev_after']-x['base_eu']  # cost removed
        best=max(candidates,key=candidates.get)
        if candidates[best]>0 and entropy(b)>0.8:
            test=best; test_pos=sample_test(test,true,rng); b=update_binary_test(b,test,test_pos)
        action,human,_=pomdp_action(b)
    elif ab=='AB3':
        # POMDP with cost-aware tests but no explicit positive-VOI gate: test if gross gain exceeds half the cost.
        v={t:voi(b,t) for t in tests}; best=max(v,key=lambda t:v[t]['voi'])
        if (v[best]['ev_after']-v[best]['base_eu']) > tests[best]['cost']/2 and entropy(b)>0.8:
            test=best; test_pos=sample_test(test,true,rng); b=update_binary_test(b,test,test_pos)
        action,human,_=pomdp_action(b)
    elif ab=='AB4':
        d=bdk_action_before_test(b)
        if d.get('test'):
            test=d['test']; test_pos=sample_test(test,true,rng); b=update_binary_test(b,test,test_pos)
        action,human,_=pomdp_action(b)
        human=False  # No human-escalation gate in AB4.
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

# POMDP scenario summary (single deterministic decision with the most likely state and no sampled test: M5).
pomdp_rows=[]
for sc in scenarios:
    b=scenario_beliefs[sc]; a,h,reason=pomdp_action(b); true=scenarios[sc]['true']
    pomdp_rows.append([sc,b['S0'],b['S1'],b['S2'],b['S3'],a,action_name[a],true,'Evet' if is_correct_decision(true,a,h) else 'Hayir',reason])

# VOI-POMDP deterministic plan summary before the actual test result.
bdk_rows=[]
for sc,b in scenario_beliefs.items():
    d=bdk_action_before_test(b)
    bdk_rows.append([sc,b['S0'],b['S1'],b['S2'],b['S3'],d.get('test') or '',d.get('action') or '',d.get('reason','')])

# Output helpers

def write_csv(filename,headers,rows):
    with open(OUT/filename,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(headers); w.writerows(rows)

write_csv('prior_check.csv',['State','Probability','Contribution_to_Sum'],[[s,prior[s],prior[s]] for s in active_states]+[['Total',sum(prior[s] for s in active_states),1.0]])
write_csv('bayes_o1.csv',['State','Prior','P(O1|S)','Product','Posterior'],[[s,prior[s],obs_prob[s]['O1'],numer[s],o1_post[s]] for s in active_states]+[['Total','','',den,1.0]])
write_csv('belief_updates.csv',['Scenario','Time','Observation','P(S0)','P(S1)','P(S2)','P(S3)','En_Olasi_State'],belief_rows)
write_csv('expected_utility.csv',['Scenario','Action','Action_Name','Expected_Security_Gain','Intervention_Cost','False_Intervention_Penalty','Delay_Penalty','Expected_Utility'],utility_rows)
write_csv('value_of_information.csv',['Scenario','Test','Test_Name','Best_Utility_Without_Test','Expected_Utility_After_Test','Test_Cost','Value_of_Information','Decision'],voi_rows)
write_csv('mdp_policy.csv',['State','State_Name','Action','Action_Name','One_Step_EU','MDP_Q_Value','Reason'],mdp_rows)
write_csv('pomdp_results.csv',['Scenario','P(S0)','P(S1)','P(S2)','P(S3)','Secilen_Action','Action_Name','Gercek_State','Dogru_Decision','Reason'],pomdp_rows)
write_csv('bdk_pomdp_results.csv',['Scenario','P(S0)','P(S1)','P(S2)','P(S3)','Selected_Test','Dogrudan_Action','Reason'],bdk_rows)
write_csv('raw_llm_decisions.csv',['Scenario','Action','Action_Name','LLM_Attack_Probability','Reason','Probability_Verdi','False_Certainty','Cost_Consideration','Error_Codes'],llm_rows)
write_csv('experiment_540_runs.csv',list(run_records[0].keys()),[list(r.values()) for r in run_records])
write_csv('general_metrics.csv',['Model','OSO','YIO','DKO','IYO','OMG','BTO','YBTO','Brier','Calibration_Score','Delay_Success','MGS','Level'],metric_rows)
write_csv('ablation_results.csv',['Configuration','Description','OSO','YIO','DKO','Test_Cost','OMG','MGS'],ab_rows)

# Evaluation answers, based on the generated results.
best_mgs=max(metric_map,key=lambda m:metric_map[m]['MGS'])
best_deploy=max([m for m in models if m!='M4'],key=lambda m:metric_map[m]['MGS'])
highest_yio=max(metric_map,key=lambda m:metric_map[m]['YIO'])
highest_oso=max(metric_map,key=lambda m:metric_map[m]['OSO'])
# Positive/negative VOI scenarios and tests based on deterministic analysis.
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
        'Since the document does not provide a numerical coefficient for the delay penalty, the delay penalty was set to 0 in one-step expected-utility calculations; delay was measured through MDP/POMDP transitions and OMG.',
        'Information tests were modeled as binary attack/normal tests; sensitivity was applied to S1-S3 and specificity to S0.',
        'Because no separate CPT was provided for the M2 Bayesian Network, a dynamic Bayesian network (S_t -> S_t+1 -> O_t) was used with the transitions in Table 7.',
        'MGS gecikme basarisi icin maksimum izinli adim dokumanda sayisal verilmediginden 5 adim selected (en uzun 4 gozlem + 1 ek test).',
        'Because the document does not provide an alert-scoring function for the M0 fixed threshold, the experimental assumptions O0=0, O1=1, O2=2, O3=3, and threshold=3 were used.',
        'The raw LLM probability distribution was converted for the Brier calculation using a reproducible calibration representation that allocates the recorded attack probability to the dominant attack class most consistent with the observations.'
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
