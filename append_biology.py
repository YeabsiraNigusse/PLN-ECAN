import collections

import os

base_dir = os.path.dirname(os.path.abspath(__file__))

kb_file = os.path.join(base_dir, "data", "biology_kb.metta")
incident_file = os.path.join(base_dir, "data", "ecan_pln_experiment.metta")
# Dictionary to hold the incident connections for each atom
incidents = collections.defaultdict(list)

print(f"Reading {kb_file}...")
with open(kb_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        
        # Format: ((LinkType source target) (strength confidence))
        # e.g. ((InheritanceLink ant insect) (0.90 0.90))
        parts = line.replace('(', ' ').replace(')', ' ').split()
        if len(parts) >= 5:
            link_type = parts[0].lower()
            source = parts[1].lower()
            target = parts[2].lower()
            strength = parts[3]
            confidence = parts[4]
            
            link_data = f"[[{link_type},{source},{target}],[{strength},{confidence}]]"
            
            incidents[source].append(link_data)
            incidents[target].append(link_data)

print(f"Parsed {len(incidents)} unique atoms. Appending to {incident_file}...")

# Read the existing incident_final.pl to see if it already has the biology data
# to prevent duplicate appending
with open(incident_file, 'r') as f:
    existing_content = f.read()

# Append to the actual incident file
with open(incident_file, 'a') as f:
    f.write("\n\n%% AUTO-APPENDED BIOLOGY KB DATA\n")
    for atom, links in incidents.items():
        # format: '&incidentSpace'(atom, [link1, link2, ...]).
        
        # Only append if we haven't already added this exact atom's bio connections
        
        links_str = ",".join(links)
        prolog_line = f"'&incidentSpace'('{atom}',[{links_str}]).\n"
        if prolog_line not in existing_content:
            f.write(prolog_line)

print("Done appending biology data to incident_final.pl!")


# ; today meeting topics

# ; 1 visited few codebases
#    ; 1.1 hypothesis generation
#    ; 1.2 mathimatics for ai(atp)

# ; 2 ask them any suggestion including if there is existing pln-ecan experiment codebase

# ; 3 optional - show them the current trial
# ; 4 go the fluid dynamic implementation



# next task related with experiment could be 

# 1 looking in detail how the backward chainer works and where and why exactlly the combinatorial explosion occurs
# 2 investigate if ecan can mitigate this combinatorial explosion
# 3 pick a project and apply the solution found in step 2

