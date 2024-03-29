#!/usr/bin/env python3

# author: Michael A. Perlin (github.com/perlinm) , modified by Ji Liu (ji.liu@anl.gov)
#modified by Peiyi Li(pli11@ncsu.edu): remove unused functions

# number of fragments
def get_frag_num(wire_path_map):
    return len(set( frag_wire[0] for path in wire_path_map.values()
                    for frag_wire in path ))

# identify all stitches in a cut-up circuit, in dictionary format:
#   { <exit wire> : <init wire> }
def identify_stitches(wire_path_map):
    circuit_wires = list(wire_path_map.keys())
    stitches = {}
    for wire in circuit_wires:
        # identify all init/exit wires in the path of this wire
        init_wires = wire_path_map[wire][1:]
        exit_wires = wire_path_map[wire][:-1]
        # add the stitches in this path
        stitches.update({ exit_wire : init_wire
                          for init_wire, exit_wire in zip(init_wires, exit_wires) })
    return stitches

# identify preparation / meauserment qubits for all fragments
def identify_frag_targets(wire_path_map):
    stitches = identify_stitches(wire_path_map)
    frag_targets = [ { "meas" : tuple(), "prep" : tuple() }
                     for _ in range(get_frag_num(wire_path_map)) ]
    for meas_frag_qubit, prep_frag_qubit in stitches.items():
        meas_frag, meas_qubit = meas_frag_qubit
        prep_frag, prep_qubit = prep_frag_qubit
        frag_targets[meas_frag]["meas"] += (meas_qubit,)
        frag_targets[prep_frag]["prep"] += (prep_qubit,)
    return frag_targets

