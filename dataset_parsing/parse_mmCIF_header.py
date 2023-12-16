"""
Prody seems to struggle to parse the header of some mmCIF files,
particularly for the biological assembly mmCIF files that I want to use to generate my dataset.
This module is a minimal version of the parse_header reconstructing basic functionality from Prody's normal MMCIFHeader parsing.

Benjamin Fry (bfry@g.harvard.edu)
"""

from prody import parseSTARSection, Polymer, Chemical, DBRef 
from prody.atomic import flags, ATOMIC_FIELDS

from collections import defaultdict, OrderedDict
import numpy as np
from typing import Optional, Tuple
import io


_PDB_DBREF = {
    'GB': 'GenBank',
    'PDB': 'PDB',
    'UNP': 'UniProt',
    'NORINE': 'Norine',
    'UNIMES': 'UNIMES',
    'EMDB': 'EMDB',
    'BMRB': 'BMRB'
}

_COMPND_KEY_MAPPINGS = {'_entity.id': 'MOL_ID',
                        '_entity.pdbx_description': 'MOLECULE',
                        '_entity.pdbx_fragment': 'FRAGMENT',
                        '_entity_name_com.name': 'SYNONYM',
                        '_entity.pdbx_ec': 'EC',
                        '_entity.pdbx_mutation': 'MUTATION',
                        '_entity.details': 'OTHER_DETAILS'}


def cleanString(string, nows=False):
    """*nows* is no white space."""
    if nows:
        return ''.join(string.strip().split())
    else:
        return ' '.join(string.strip().split())

def _natomsFromFormulaPart(part):
    digits = [s for s in part if s.isdigit()]
    if len(digits) == 0:
        return 1
    return int("".join(digits))


def _natomsFromFormula(formula, hydrogens=False):
    if ("(" in formula and ")" in formula):
        formula = formula.split("(")[1].split(")")[0]

    parts = formula.split()
    if not hydrogens:
        parts = [part for part in parts if part.find("H") == -1]

    return sum([_natomsFromFormulaPart(part) for part in parts])


def _getPolymers(lines, pdbid=None):
    """Returns list of ProDy polymers (macromolecules)."""

    pdbid = "XXXX"
    if pdbid is not None:
        pdbid = pdbid
    polymers = dict()

    entities = defaultdict(list)

    # SEQRES block
    items1 = parseSTARSection(lines, '_entity_poly')

    for item in items1:
        chains = item['_entity_poly.pdbx_strand_id']
        entity = item['_entity_poly.entity_id']

        for ch in chains.split(","):
            entities[entity].append(ch)
            poly = polymers.get(ch, Polymer(ch))
            polymers[ch] = poly
            poly.sequence += ''.join(item[
                '_entity_poly.pdbx_seq_one_letter_code_can'].replace(';', '').split())

    # DBREF block 1
    items2 = parseSTARSection(lines, '_struct_ref')

    for item in items2:
        entity = item["_struct_ref.id"]
        chains = entities[entity]
        for ch in chains:
            dbabbr = item["_struct_ref.db_name"]
            dbref = DBRef()
            dbref.dbabbr = dbabbr
            dbref.database = _PDB_DBREF.get(dbabbr, 'Unknown')
            dbref.accession = item["_struct_ref.pdbx_db_accession"]
            dbref.idcode = item["_struct_ref.db_code"]

            poly = polymers[ch]
            poly.dbrefs.append(dbref)

    # DBREF block 2
    items3 = parseSTARSection(lines, "_struct_ref_seq")

    for i, item in enumerate(items3):
        i += 1

        ch = item["_struct_ref_seq.pdbx_strand_id"]
        poly = polymers[ch] 
        
        for dbref in poly.dbrefs:
            if item["_struct_ref_seq.pdbx_db_accession"] == dbref.accession:

                try:
                    first = int(item["_struct_ref_seq.pdbx_auth_seq_align_beg"])
                    initICode = item["_struct_ref_seq.pdbx_db_align_beg_ins_code"]
                    if initICode == '?':
                        initICode = ' '
                except:
                    try:
                        first = int(item["_struct_ref_seq.pdbx_auth_seq_align_beg"])
                        initICode = item["_struct_ref_seq.pdbx_seq_align_beg_ins_code"]
                        if initICode == '?':
                            initICode = ' '
                    except:
                        print('DBREF for chain {2}: failed to parse '
                                    'initial sequence number of the PDB sequence '
                                    '({0}:{1})'.format(pdbid, i, ch))
                try:
                    last = int(item["_struct_ref_seq.pdbx_auth_seq_align_end"])
                    endICode = item["_struct_ref_seq.pdbx_db_align_end_ins_code"]
                    if endICode == '?':
                        endICode = ' '
                except:
                    try:
                        last = int(item["_struct_ref_seq.pdbx_auth_seq_align_end"])
                        endICode = item["_struct_ref_seq.pdbx_seq_align_end_ins_code"]
                        if endICode == '?':
                            endICode = ' '
                    except:
                        print('DBREF for chain {2}: failed to parse '
                                    'ending sequence number of the PDB sequence '
                                    '({0}:{1})'.format(pdbid, i, ch))
                try:
                    first2 = int(item["_struct_ref_seq.db_align_beg"])
                    dbref.first = (first, initICode, first2)
                except:
                    print('DBREF for chain {2}: failed to parse '
                                'initial sequence number of the database sequence '
                                '({0}:{1})'.format(pdbid, i, ch))
                try:
                    last2 = int(item["_struct_ref_seq.db_align_end"])
                    dbref.last = (last, endICode, last2)
                except:
                    print('DBREF for chain {2}: failed to parse '
                                'ending sequence number of the database sequence '
                                '({0}:{1})'.format(pdbid, i, ch))

    for poly in polymers.values():  # PY3K: OK
        resnum = []
        for dbref in poly.dbrefs:
            dbabbr = dbref.dbabbr
            if dbabbr == 'PDB':
                if not (pdbid == dbref.accession == dbref.idcode):
                    print('DBREF for chain {2} refers to PDB '
                                'entry {3} ({0}:{1})'
                                .format(pdbid, i, ch, dbref.accession))
            else:
                if pdbid == dbref.accession or pdbid == dbref.idcode:
                    print('DBREF for chain {2} is {3}, '
                                'expected PDB ({0}:{1})'
                                .format(pdbid, i, ch, dbabbr))
                    dbref.database = 'PDB'
            
            try:
                resnum.append((dbref.first[0], dbref.last[0]))
            except:
                pass # we've already warned about this

        resnum.sort()
        last = -10000
        for first, temp in resnum:
            if first <= last:
                print('DBREF records overlap for chain {0} ({1})'
                            .format(poly.chid, pdbid))
            last = temp

    # MODRES block
    data4 = parseSTARSection(lines, "_pdbx_struct_mod_residue")

    for data in data4:
        ch = data["_pdbx_struct_mod_residue.label_asym_id"]

        poly = polymers.get(ch, Polymer(ch))
        polymers[ch] = poly
        if poly.modified is None:
            poly.modified = []

        iCode = data["_pdbx_struct_mod_residue.PDB_ins_code"]
        if iCode == '?':
            iCode == '' # PDB one is stripped
        poly.modified.append((data["_pdbx_struct_mod_residue.auth_comp_id"],
                                data["_pdbx_struct_mod_residue.auth_asym_id"],
                                data["_pdbx_struct_mod_residue.auth_seq_id"] + iCode,
                                data["_pdbx_struct_mod_residue.parent_comp_id"],
                                data["_pdbx_struct_mod_residue.details"]))

    # SEQADV block
    data5 = parseSTARSection(lines, "_struct_ref_seq_dif")

    for i, data in enumerate(data5):
        ch = data["_struct_ref_seq_dif.pdbx_pdb_strand_id"]

        poly = polymers.get(ch, Polymer(ch))
        polymers[ch] = poly
        dbabbr = data["_struct_ref_seq_dif.pdbx_seq_db_name"]
        resname = data["_struct_ref_seq_dif.mon_id"]
        if resname == '?':
            resname = '' # strip for pdb

        try:
            resnum = int(data["_struct_ref_seq_dif.pdbx_auth_seq_num"])
        except:
            #print('SEQADV for chain {2}: failed to parse PDB sequence '
            #            'number ({0}:{1})'.format(pdbid, i, ch))
            continue

        icode = data["_struct_ref_seq_dif.pdbx_pdb_ins_code"]
        if icode == '?':
            icode = '' # strip for pdb            
        
        try:
            dbnum = int(data["_struct_ref_seq_dif.pdbx_seq_db_seq_num"])
        except:
            #print('SEQADV for chain {2}: failed to parse database '
            #            'sequence number ({0}:{1})'.format(pdbid, i, ch))
            continue            

        comment = data["_struct_ref_seq_dif.details"].upper()
        if comment == '?':
            comment = '' # strip for pdb 

        match = False
        for dbref in poly.dbrefs:
            if not dbref.first[0] <= resnum <= dbref.last[0]:
                continue
            match = True
            if dbref.dbabbr != dbabbr:
                print('SEQADV for chain {2}: reference database '
                            'mismatch, expected {3} parsed {4} '
                            '({0}:{1})'.format(pdbid, i+1, ch,
                            repr(dbref.dbabbr), repr(dbabbr)))
                continue
            dbacc = data["_struct_ref_seq_dif.pdbx_seq_db_accession_code"]
            if dbref.accession[:9] != dbacc[:9]:
                print('SEQADV for chain {2}: accession code '
                            'mismatch, expected {3} parsed {4} '
                            '({0}:{1})'.format(pdbid, i+1, ch,
                            repr(dbref.accession), repr(dbacc)))
                continue
            dbref.diff.append((resname, resnum, icode, dbnum, dbnum, comment))
        if not match:
            print('SEQADV for chain {2}: database sequence reference '
                        'not found ({0}:{1})'.format(pdbid, i+1, ch))
            continue

    # COMPND double block. 
    # Block 6 has most info. Block 7 has synonyms
    data6 = parseSTARSection(lines, "_entity")
    data7 = parseSTARSection(lines, "_entity_name_com")

    dict_ = {}
    for molecule in data6:
        dict_.clear()
        for k, value in molecule.items():
            if k == '_entity.id':
                dict_['CHAIN'] = ', '.join(entities[value])

            try:
                key = _COMPND_KEY_MAPPINGS[k]
            except:
                continue
            val = value.strip()
            if val == '?':
                val = ''
            dict_[key.strip()] = val

        chains = dict_.pop('CHAIN', '').strip()

        if not chains:
            continue
        for ch in chains.split(','):
            ch = ch.strip()
            poly = polymers.get(ch, Polymer(ch))
            polymers[ch] = poly
            poly.name = dict_.get('MOLECULE', '').upper()

            poly.fragment = dict_.get('FRAGMENT', '').upper()

            poly.comments = dict_.get('OTHER_DETAILS', '').upper()

            val = dict_.get('EC', '')
            poly.ec = [s.strip() for s in val.split(',')] if val else []

            poly.mutation = dict_.get('MUTATION', '') != ''
            poly.engineered = dict_.get('ENGINEERED', poly.mutation)

    for molecule in data7:
        dict_.clear()
        for k, value in molecule.items():
            if k.find('entity_id') != -1:
                dict_['CHAIN'] = ', '.join(entities[value])

            try:
                key = _COMPND_KEY_MAPPINGS[k]
            except:
                continue
            dict_[key.strip()] = value.strip()

        chains = dict_.pop('CHAIN', '').strip()

        if not chains:
            continue
        for ch in chains.split(','):
            ch = ch.strip()
            poly = polymers.get(ch, Polymer(ch))
            polymers[ch] = poly

            val = dict_.get('SYNONYM', '')
            poly.synonyms = [s.strip().upper() for s in val.split(',')
                                ] if val else []

    return list(polymers.values())


def _getChemicals(lines):
    """Returns list of chemical components (heterogens)."""

    chemicals = defaultdict(list)
    chem_names = defaultdict(str)
    chem_synonyms = defaultdict(str)
    chem_formulas = defaultdict(str)
    chem_n_atoms = defaultdict(int)

    # Data is split across blocks again
    # 1st block we need is has info about location in structure
    # this instance only includes single sugars not branched structures
    items = parseSTARSection(lines, "_pdbx_nonpoly_scheme")

    for data in items:
        resname = data["_pdbx_nonpoly_scheme.mon_id"]
        if resname in flags.AMINOACIDS or resname == "HOH":
            continue

        chem = Chemical(resname)
        chem.chain = data["_pdbx_nonpoly_scheme.pdb_strand_id"]
        chem.resnum = int(data["_pdbx_nonpoly_scheme.pdb_seq_num"])

        icode = data["_pdbx_nonpoly_scheme.pdb_ins_code"]
        if icode == '.':
            icode = ''
        chem.icode = icode
        chem.description = '' # often empty in .pdb and not clearly here
        chemicals[chem.resname].append(chem)

    # next we get the equivalent one for branched sugars part
    items = parseSTARSection(lines, "_pdbx_branch_scheme")

    for data in items:
        resname = data["_pdbx_branch_scheme.mon_id"]
        if resname in flags.AMINOACIDS or resname == "HOH":
            continue

        chem = Chemical(resname)
        chem.chain = data["_pdbx_branch_scheme.pdb_asym_id"]
        chem.resnum = int(data["_pdbx_branch_scheme.pdb_seq_num"])

        chem.icode = '' # this part doesn't have this field
        chem.description = '' # often empty in .pdb and not clearly here
        chemicals[chem.resname].append(chem)

    # 2nd block to get has general info e.g. name and formula
    items = parseSTARSection(lines, "_chem_comp")

    for data in items:
        resname = data["_chem_comp.id"]
        if resname in flags.AMINOACIDS or resname == "HOH":
            continue

        chem_names[resname] += data["_chem_comp.name"].upper()

        if "_chem_comp.pdbx_synonyms" in data.keys():
            synonym = data["_chem_comp.pdbx_synonyms"]
        else:
            synonym = '?'

        if synonym == '?':
            synonym = ' '
        synonym = synonym.rstrip()
        if synonym.startswith(';') and synonym.endswith(';'):
            synonym = synonym[1:-1]
        chem_synonyms[resname] += synonym
        
        chem_formulas[resname] += data["_chem_comp.formula"]


    for key, name in chem_names.items():  # PY3K: OK
        name = cleanString(name)
        for chem in chemicals[key]:
            chem.name = name

    for key, formula in chem_formulas.items():  # PY3K: OK
        formula = cleanString(formula)
        repeats = len(chemicals[key])
        for chem in chemicals[key]:
            chem.formula = '{0}({1})'.format(repeats, formula)
            chem.natoms = _natomsFromFormula(formula)

    for key, synonyms in chem_synonyms.items():  # PY3K: OK
        synonyms = cleanString(synonyms)
        synonyms = synonyms.split(';')
        for chem in chemicals[key]:
            if synonyms != ['']:
                chem.synonyms = [syn.strip() for syn in synonyms]

    alist = []
    for chem in chemicals.values():  # PY3K: OK
        for chem in chem:
            alist.append(chem)
    return alist


def map_seqres_index(lines, header_chids):
    """
    Parses the mmCIF chain information analogously to prody parseMMCIFStream and extracts mapping of (seg, chain, resnum) to SEQRES index.

    Args:
        lines (list): List of lines containing the mmCIF data.
        header_chids (list): List of chain IDs with sequence information to filter down the number of chains to collect data for.

    Returns:
        dict: A dictionary mapping (segnames, chainids, resnums) to seqres_index_resnums.
    """

    i = 0
    asize = 0
    start = 0
    stop = 0
    fields = OrderedDict()
    fieldCounter = -1
    foundAtomBlock = False
    doneAtomBlock = False

    while not doneAtomBlock:
        line = lines[i]
        if line[:11] == '_atom_site.':
            fieldCounter += 1
            fields[line.split('.')[1].strip()] = fieldCounter
   
        if line.startswith('ATOM ') or line.startswith('HETATM'):
                if not foundAtomBlock:
                    foundAtomBlock = True
                    start = i
                asize += 1
        else:
            if foundAtomBlock:
                doneAtomBlock = True
                stop = i
        i += 1
    
    segnames = np.zeros(asize, dtype=ATOMIC_FIELDS['segment'].dtype)
    chainids = np.zeros(asize, dtype=ATOMIC_FIELDS['chain'].dtype)
    resnums = np.zeros(asize, dtype=ATOMIC_FIELDS['resnum'].dtype)
    seqres_index_resnums = np.zeros(asize, dtype=ATOMIC_FIELDS['segment'].dtype)

    acount = 0
    for line in lines[start:stop]:
        linesplit = line.split()
        startswith = linesplit[fields['group_PDB']]
        chid = linesplit[fields['auth_asym_id']]
        chainids[acount] = chid

        if chid not in header_chids:
            continue

        segnames[acount] = linesplit[fields['label_asym_id']]
        resnums[acount] = linesplit[fields['auth_seq_id']]
        seqres_index_resnums[acount] = linesplit[fields['label_seq_id']]

        acount += 1

    return {(a,b,c): y for a,b,c,y in zip(segnames, chainids, resnums, seqres_index_resnums)}
    
def parse_mmcif_metadata(filestream: io.StringIO, pdbid: Optional[str] = None) -> Tuple[dict, dict]:
    """
    Parses an mmCIF file for features of interest.

    Args:
        filestream (io.StringIO): The filestream object containing the mmCIF file.
        pdbid (Optional[str]): The PDB ID of the file. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], Dict[str, int]]: A tuple containing the parsed header information and the sequence
        mapping dictionary.

    """
    # Resets pointer to beginning of file, reads all lines into list.
    filestream.seek(0)
    all_lines = [line for line in filestream.readlines()]

    output = {}
    output['polymers'] = _getPolymers(all_lines, pdbid)
    output['chemicals'] = _getChemicals(all_lines)
    seqres_map = map_seqres_index(all_lines, [polymer.chid for polymer in output['polymers']])
    return output, seqres_map
