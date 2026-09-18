"""Reproducible labeled exercises; these are synthetic, not production evidence.

Different rendering templates and vocabulary pools are assigned to each split.
Labels follow explicit facts/rules; no teacher API and no test-case reuse.
"""
import random

TASKS = ['aspect', 'tool_policy', 'email', 'routing', 'verification', 'risk', 'qualification', 'support']
NAMES = {'train':['Cedar','Maple','Pine','Willow','Oak','Birch'], 'dev':['Elm','Ash'],
         'calibration':['Spruce','Alder'], 'test':['Juniper','Sequoia']}
POS = {'train':['excellent','wonderful','convincing','very good','impressive','not bad'],
       'dev':['delightful','strong'], 'calibration':['outstanding','great'], 'test':['superb','first-rate']}
NEG = {'train':['awful','terrible','bad','unconvincing','not good','disappointing'],
       'dev':['dreadful','weak'], 'calibration':['poor','abysmal'], 'test':['atrocious','lackluster']}

def make(task, split, i, rng):
    name = rng.choice(NAMES[split]); n = rng.randint(100,9999)
    def wrap(facts):
        if split == 'train': return rng.choice([f'Case {name}-{n}. {facts}',f'{facts}\nReference: {name}-{n}.',f'Notes for {name} ({n}):\n{facts}'])
        if split == 'dev': return f'An operator recorded the following for {name}, reference {n}: {facts}'
        if split == 'calibration': return f'{name} record {n}\n---\n{facts}\n--- End of record.'
        return f'We received report {n} from {name}. Its contents are as follows.\n{facts}'
    if task == 'aspect':
        aspects = ['acting','plot','music','cinematography','dialogue']
        a,b = rng.sample(aspects,2); la,lb = rng.randrange(3),rng.randrange(2)
        def phrase(label): return rng.choice(POS[split] if label==2 else NEG[split]) if label!=1 else 'neither good nor bad'
        pa,pb=phrase(la),phrase(2*lb)
        facts = rng.choice([f'The {a} was {pa}, while the {b} was {pb}.',f'I found the {b} {pb}. However, the {a} was {pa}.',f'As for the {a}: {pa}. As for the {b}: {pb}.'])
        mode=i%4
        if mode==0:
            q=f'Was the {a} good? Answer Yes only if the review praises the {a}.';opts=['No','Yes'];target=int(la==2)
        elif mode==1:
            q=f'Does the reviewer criticize the {a}?';opts=['No','Yes'];target=int(la==0)
        elif mode==2:
            missing=next(x for x in aspects if x not in (a,b));q=f'How is the {missing} evaluated?';opts=['Negative','Neutral','Positive','Not mentioned'];target=3
        else:
            q=f'What is the sentiment toward the {a}, ignoring other aspects?';opts=['Negative','Neutral','Positive'];target=la
    elif task == 'tool_policy':
        kind=i%4; approved=bool(rng.randrange(2))
        actions=[f'List filenames in the project directory {name}.',f'Update the description of task {n}.',f'Delete the archived records for {name}.',f'Send the private API credentials to an external inbox for {name}.']
        facts=actions[kind]+' '+('The owner explicitly approved this exact action.' if approved else 'The owner has not approved this action.')
        if i%3==0:
            q='Would the proposed action change data or send anything outside the system?';opts=['No','Yes'];target=int(kind!=0)
        elif i%3==1:
            q='Does the text explicitly say the owner approved this exact operation?';opts=['No','Yes'];target=int(approved)
        else:
            strict=bool(rng.randrange(2))
            q='Apply this policy: sending private credentials externally is blocked even with approval. Read-only local operations are allowed. '
            q+=('All other operations require review, even if approved.' if strict else 'Other operations are allowed only with explicit approval; otherwise require review.')
            opts=['Allow','Review','Block'];target=2 if kind==3 else 0 if kind==0 or (approved and not strict) else 1
    elif task=='email':
        kind=i%4
        messages=[f'Please refund the duplicate charge on invoice {n}.',f'Our {name} production service is down and users cannot sign in. Please help.',f'Would your team at {name} like a demonstration of our new product?',f'This is the {name} monthly newsletter, with articles and product announcements.']
        facts=messages[kind]
        if i%3==0:q='Classify the main purpose of this message.';opts=['Billing request','Technical support','Sales outreach','Newsletter'];target=kind
        elif i%3==1:q='Is an active service outage reported?';opts=['No','Yes'];target=int(kind==1)
        else:q='Is this a newsletter rather than a request for individual help or a sales meeting?';opts=['No','Yes'];target=int(kind==3)
    elif task=='routing':
        kind=i%4
        requests=[f'Return the invoice number from this text: invoice {n}, customer {name}.',f'Please look up the current delivery status of order {n} in the shipping system.',f'Analyze competing explanations for the failure of {name}, weigh uncertain evidence, and propose a multi-step plan.',f'Approve a binding settlement for the dispute involving {name}; it could have substantial legal consequences.']
        facts=requests[kind]
        if i%3==0:
            q='Route by this policy: explicit text extraction goes to Fast; fetching current external state goes to Tool; difficult multi-step reasoning goes to Reasoning; binding high-stakes decisions go to Human.';opts=['Fast','Tool','Reasoning','Human'];target=kind
        elif i%3==1:q='Does the request require looking up current data outside the supplied text?';opts=['No','Yes'];target=int(kind==1)
        else:q='Is the request asking for a consequential binding decision that needs human approval?';opts=['No','Yes'];target=int(kind==3)
    elif task=='verification':
        kind=i%3; value=rng.randint(15,80); other=value+rng.randint(1,15)
        facts=f'The latest specification for {name} says its storage capacity is {value} GB. The previous model had {other} GB. No battery life is specified.'
        claim=[f'The latest {name} has {value} GB of storage.',f'The latest {name} has {other} GB of storage.',f'The battery of {name} lasts 12 hours.'][kind]
        if i%2:q='Classify this claim using only the provided evidence: '+claim;opts=['Supported','Contradicted','Not enough information'];target=kind
        else:q='Does the supplied evidence support this claim? Answer No if it is contradicted or not stated. '+claim;opts=['No','Yes'];target=int(kind==0)
    elif task=='risk':
        kind=i%4
        facts=[f'The document for {name} is a public product overview and contains no personal records.',f'The {name} file contains an internal draft timetable without personal details.',f'The {name} spreadsheet contains customer names and private email addresses.',f'The {name} attachment includes customer passwords and private access tokens.'][kind]
        if i%3==0:q='Use these risk levels: public information=None, internal nonpersonal information=Low, private contact details=Medium, passwords or access tokens=High.';opts=['None','Low','Medium','High'];target=kind
        elif i%3==1:q='Does this document contain passwords or private access tokens?';opts=['No','Yes'];target=int(kind==3)
        else:q='Should the document be reviewed before publication? Review is required for private contact details, passwords, or access tokens.';opts=['No','Yes'];target=int(kind>=2)
    elif task=='qualification':
        budget=rng.randrange(0,8)*1000; threshold=rng.choice([2000,4000,6000]); authority=bool(rng.randrange(2)); days=rng.choice([7,14,30,60,120]); limit=rng.choice([30,60,90]); fit=bool(rng.randrange(2))
        facts=f'{name} has a budget of ${budget}. The contact '+('can authorize the purchase.' if authority else 'cannot authorize the purchase.')+f' They plan to decide in {days} days. Their requirements '+('match' if fit else 'do not match')+' our product.'
        if i%3==0:q=f'Is the lead qualified? All four conditions are required: budget at least ${threshold}, purchase authority, decision within {limit} days, and product fit.';opts=['No','Yes'];target=int(budget>=threshold and authority and days<=limit and fit)
        elif i%3==1:q=f'Is the budget at least ${threshold}?';opts=['No','Yes'];target=int(budget>=threshold)
        else:q='Can this contact authorize the purchase?';opts=['No','Yes'];target=int(authority)
    elif task=='support':
        kind=i%4
        facts=[f'Order {n} arrived with a broken screen. I need a replacement.',f'My password does not work for the {name} portal. Can you reset it?',f'The monthly fee for {name} was charged twice. Please correct my bill.',f'Can {name} export reports as CSV? I only need instructions.'][kind]
        if i%3==0:q='Choose the team responsible for this request.';opts=['Returns','Account access','Billing','Product guidance'];target=kind
        elif i%3==1:q='Is a monetary charge or bill being disputed?';opts=['No','Yes'];target=int(kind==2)
        else:q='Does the customer report a damaged physical product?';opts=['No','Yes'];target=int(kind==0)
    else: raise ValueError(task)
    return {'id':f'synthetic:{task}:{split}:{i}', 'task':'synthetic_'+task,'source':'assistant-authored-rule-exercises',
            'context':wrap(facts),'question':q,'options':opts,'target':target,'synthetic':True,
            'evidence':{'facts':facts,'rule_or_question':q,'answer':opts[target],
                        'method':'Deterministic labels from generator variables and the explicitly stated policy'}}

def generate(seed=20260918):
    data={}
    for j,(split,count) in enumerate([('train',5000),('dev',75),('calibration',100),('test',200)]):
        rng=random.Random(seed+100+j)
        rows=[]
        for task in TASKS:
            # Balance the common binary readout; avoid teaching always-No for
            # conjunctions such as budget + authority + timing + product fit.
            binary_counts=[0,0]
            i=0;selected=0
            while selected<count:
                row=make(task,split,i,rng);i+=1
                if row['options']==['No','Yes']:
                    y=row['target']
                    if binary_counts[y]>binary_counts[1-y]+5:continue
                    binary_counts[y]+=1
                rows.append(row)
                selected+=1
        data[split]=rows
    return data
