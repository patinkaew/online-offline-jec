def listOut(*recs):
    for r in recs: print(r)
    print()

print()
listOut('Incunabula', 'Amber', 'Tri Repetae')
trilogy = 'Incunabula', 'Amber', 'Tri Repetae'
listOut(trilogy)    # Reads as just one argument
listOut(*trilogy)   # Reads as tuple

WARP128 = {
    'name': 'Confield',
    'artist': 'Autechre',
    'year': 2001
}

print(WARP128)

def readDict(**rec):
    for key, value in rec.items(): print(key, '|', value)
    print()

readDict(**WARP128)

# Should be switched order:
# dataset_name = dataset_name if dataset_name != "*" else get_default_dataset_name(filelist[0])
# filelist = get_filelist(input_path)