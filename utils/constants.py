"""Holds all the constants shared accross the project"""
import pathlib
import unittest

# All constants shared accross models are defined here
BACKUPFOLDERPATH  = pathlib.Path(r'D:\MAPHIS')
RDSFFOLDERPATH    = pathlib.Path(r'\\rdsfcifs.acrc.bris.ac.uk\MAPHIS_historical_maps')
DATASETFOLDERPATH = BACKUPFOLDERPATH.joinpath('datasets')
IMAGESFOLDERPATH  = DATASETFOLDERPATH.joinpath('images')
CLASSIFIEDPATH    = DATASETFOLDERPATH.joinpath('classified')
CITIESFOLDERPATH  = DATASETFOLDERPATH.joinpath('cities')
RAWPATH           = DATASETFOLDERPATH.joinpath('raw')
PROCESSEDPATH     = DATASETFOLDERPATH.joinpath('processed')
GRAPHPATH         = DATASETFOLDERPATH.joinpath('graphs')
MODELSPATH        = DATASETFOLDERPATH.joinpath('models')
TRAININGPATH      = DATASETFOLDERPATH.joinpath('training')
TILESDATAPATH     = DATASETFOLDERPATH.joinpath('tiles_data')
SHAPESDATAPATH    = DATASETFOLDERPATH.joinpath('shape_files')
PROJECTPATHS = [
    BACKUPFOLDERPATH,
    RDSFFOLDERPATH,
    DATASETFOLDERPATH,
    IMAGESFOLDERPATH,
    CLASSIFIEDPATH,
    CITIESFOLDERPATH,
    RAWPATH,
    PROCESSEDPATH,
    GRAPHPATH,
    MODELSPATH,
    TRAININGPATH,
    TILESDATAPATH,
    SHAPESDATAPATH
    ]
FILEEXTENSION = '.jpg'

# Features Names
FEATURENAMES = ('buildings', 'embankments', 'labels', 'neighbourhoods',
                'rail', 'rivers', 'stamps_large_font', 'stamps_small_font', 'trees')

# Hex keys
HEXKEYS = {
    'buildings':'#fe0000',
    'embankments':'#9a7b4f',
    'labels':'#1e74fd',
    'neighbourhoods':'#ffff00',
    'rail':'#6a0dad',
    'rivers':'#000080',
    'stamps_large_font':'#00ffff',
    'stamps_small_font':'#ffa500',
    'trees':'#1dfd4d',
    }

CLASSESFILES = {
    'stamps_large_font': CLASSIFIEDPATH.joinpath(r'stamps_large_font\classes.json'),
    'stamps_small_font': CLASSIFIEDPATH.joinpath(r'stamps_small_font\classes.json')
}

# Color thresholds when converted to gray scale
COLORTHRESHOLD = {
    'buildings':77,
    'embankments':127,
    'labels':106,
    'neighbourhoods':227,
    'rail':59,
    'rivers':15,
    'stamps_large_font':179,
    'stamps_small_font':174,
    'trees':167,
    }

HIGHLEVELFEATURES = ['imprint', 'text','vegetation']

HIGHTOLOWDETECTION = {
    'imprint':['buildings'],
    'embankments':['embankments'],
    'rail':['rail'],
    'rivers':['rivers'],
    'text':['labels', 'stamps_small_font', 'stamps_large_font'],
    'vegetation':['trees']
    }

LOWTOHIGHTDETECTION= {
    'buildings':'imprint',
    'labels':'text',
    'stamps_large_font':'text',
    'stamps_small_font':'text',
    'trees':'vegetation'
    }

HIGHTOLOWDISPLAY = {
    'imprint':['buildings', 'embankments', 'neighbourhoods', 'rail', 'rivers'],
    'text':['labels', 'stamps_small_font', 'stamps_large_font'],
    'vegetation':['trees']
    }

LOWTOHIGHTDISPLAY = {
    'buildings':'imprint',
    'labels':'text',
    'neighbourhoods':'imprint',
    'rail':'imprint',
    'rivers':'imprint',
    'stamps_large_font':'text',
    'stamps_small_font':'text',
    'trees':'vegetation'
    }

BACKGROUNDKWDS = ['neighbourhoods', 'rail', 'rivers']

# Tile sizes
TILEHEIGHT = 7590
TILEWIDTH = 11400

# Thumbnail parameters
KERNELSIZE = 512
WIDTHPADDING=100
HEIGHTPADDING = 157
WIDTHSTRIDE = 50
HEIGHTSTRIDE = 50
NCOLS = 24
NROWS = 16

PROXIMITY = 250

CITYKEY = {
    "0": {"Town": "Barrow-in-Furness", "County": "Lancashire", "Model_flat": "CE_flat"},
    "1": {"Town": "Bedford", "County": "Bedfordshire", "Model_flat": "CE_flat"},
    "2": {"Town": "Birkenhead", "County": "Cheshire", "Model_flat": "CE_flat"},
    "3": {"Town": "Birmingham", "County": "Warwickshire", "Model_flat": "CE_flat"},
    "4": {"Town": "Blackburn", "County": "Lancashire", "Model_flat": "CE_flat"},
    "5": {"Town": "Bolton", "County": "Lancashire", "Model_flat": "CE_flat"},
    "6": {"Town": "Bradford", "County": "Yorkshire", "Model_flat": "CE_flat"},
    "7": {"Town": "Bristol", "County": "Gloucestershire", "Model_flat": "SE_flat"},
    "8": {"Town": "Burnley", "County": "Lancashire", "Model_flat": "CE_flat"},
    "9": {"Town": "Burton-upon-trent", "County": "Staffordshire", "Model_flat": "CE_flat"},
    "10": {"Town": "Cardiff", "County": "Glamorganshire", "Model_flat": "SE_flat"},
    "11": {"Town": "Carlisle", "County": "Cumberland", "Model_flat": "NE_flat"},
    "12": {"Town": "Castleford", "County": "Yorkshire", "Model_flat": "CE_flat"},
    "13": {"Town": "Rochester", "County": "Kent", "Model_flat": "SE_flat"},
    "14": {"Town": "Chester", "County": "Cheshire", "Model_flat": "CE_flat"},
    "15": {"Town": "Coventry", "County": "Warwickshire", "Model_flat": "CE_flat"},
    "16": {"Town": "Crewe", "County": "Cheshire", "Model_flat": "CE_flat"},
    "17": {"Town": "Croydon", "County": "Surrey", "Model_flat": "SE_flat"},
    "18": {"Town": "Darlington", "County": "Durham", "Model_flat": "NE_flat"},
    "19": {"Town": "Derby", "County": "Derbyshire", "Model_flat": "CE_flat"},
    "20": {"Town": "Dover", "County": "Kent", "Model_flat": "SE_flat"},
    "21": {"Town": "Gateshead", "County": "Durham", "Model_flat": "NE_flat"},
    "22": {"Town": "Gloucester", "County": "Gloucestershire", "Model_flat": "SE_flat"},
    "23": {"Town": "Grimsby", "County": "Lincolnshire", "Model_flat": "CE_flat"},
    "24": {"Town": "Halifax", "County": "Yorkshire", "Model_flat": "CE_flat"},
    "25": {"Town": "Huddersfield", "County": "Yorkshire", "Model_flat": "CE_flat"},
    "26": {"Town": "Kingston-upon-Hull", "County": "upon", "Model_flat": "Hull"},
    "27": {"Town": "Ipswich", "County": "Suffolk", "Model_flat": "EA_flat"},
    "28": {"Town": "Wallsend", "County": "Northumberland", "Model_flat": "CE_flat"},
    "29": {"Town": "Keighley", "County": "Yorkshire", "Model_flat": "CE_flat"},
    "30": {"Town": "Kidderminster", "County": "Worcestershire", "Model_flat": "CE_flat"},
    "31": {"Town": "Leeds", "County": "Yorkshire", "Model_flat": "CE_flat"},
    "32": {"Town": "Leicester", "County": "Leicestershire", "Model_flat": "CE_flat"},
    "33": {"Town": "Lincoln", "County": "Lincolnshire", "Model_flat": "CE_flat"},
    "34": {"Town": "Liverpool", "County": "Lancashire", "Model_flat": "CE_flat"},
    "35": {"Town": "London", "County": "London", "Model_flat": "SE_flat"},
    "36": {"Town": "Luton", "County": "Bedfordshire", "Model_flat": "SE_flat"},
    "37": {"Town": "Macclesfield", "County": "Cheshire", "Model_flat": "CE_flat"},
    "38": {"Town": "Manchester", "County": "Lancashire", "Model_flat": "CE_flat"},
    "39": {"Town": "MerthyrTydvil", "County": "Tydvil", "Model_flat": "Glamorganshire"},
    "40": {"Town": "Middlesbrough", "County": "Yorkshire", "Model_flat": "CE_flat"},
    "41": {"Town": "Newcastle-upon-Tyne", "County": "Northumberland", "Model_flat": "NE_flat"},
    "42": {"Town": "Newport", "County": "Monmmouthshire", "Model_flat": "SE_flat"},
    "43": {"Town": "Northampton", "County": "Northamptonshire", "Model_flat": "CE_flat"},
    "44": {"Town": "Norwich", "County": "Norfolk", "Model_flat": "EA_flat"},
    "45": {"Town": "Nottingham", "County": "Nottinghamshire", "Model_flat": "CE_flat"},
    "46": {"Town": "Oldham", "County": "Lancashire", "Model_flat": "CE_flat"},
    "47": {"Town": "Peterborough", "County": "Northamptonshire", "Model_flat": "EA_flat"},
    "48": {"Town": "Plymouth", "County": "Devon", "Model_flat": "SE_flat"},
    "49": {"Town": "Portsmouth", "County": "Hampshire", "Model_flat": "SE_flat"},
    "50": {"Town": "Preston", "County": "Lancashire", "Model_flat": "CE_flat"},
    "51": {"Town": "Reading", "County": "Berkshire", "Model_flat": "SE_flat"},
    "52": {"Town": "Rochdale", "County": "Lancashire", "Model_flat": "CE_flat"},
    "53": {"Town": "Sheerness", "County": "Kent", "Model_flat": "SE_flat"},
    "54": {"Town": "Sheffield", "County": "Yorkshire", "Model_flat": "CE_flat"},
    "55": {"Town": "Southampton", "County": "Hampshire", "Model_flat": "SE_flat"},
    "56": {"Town": "Tynemouth", "County": "Northumberland", "Model_flat": "NE_flat"},
    "57": {"Town": "Stockport", "County": "Cheshire", "Model_flat": "CE_flat"},
    "58": {"Town": "Stockton-on-Tees", "County": "Durham", "Model_flat": "NE_flat"},
    "59": {"Town": "Stoke-on-Trent", "County": "Staffordshire", "Model_flat": "CE_flat"},
    "60": {"Town": "Sunderland", "County": "Durham", "Model_flat": "NE_flat"},
    "61": {"Town": "Swansea", "County": "Glamorganshire", "Model_flat": "SE_flat"},
    "62": {"Town": "Swindon", "County": "Wiltshire", "Model_flat": "SE_flat"},
    "63": {"Town": "Taunton", "County": "Somerset", "Model_flat": "SE_flat"},
    "64": {"Town": "Walsall", "County": "Staffordshire", "Model_flat": "CE_flat"},
    "65": {"Town": "Warrington", "County": "Lancashire", "Model_flat": "CE_flat"},
    "66": {"Town": "Wigan", "County": "Lancashire", "Model_flat": "CE_flat"},
    "67": {"Town": "Wolverhampton", "County": "Staffordshire", "Model_flat": "CE_flat"},
    "68": {"Town": "Worcester", "County": "Worcestershire", "Model_flat": "SE_flat"},
    "69": {"Town": "York", "County": "Yorkshire", "Model_flat": "CE_flat"},
    "70": {"Town": "Demo", "County": "Demo", "Model_flat": "CE_flat"}
    }

CARDINALTOCARTESIAN = {
            'north':(-1,0),
            'north_east':(-1,1),
            'east':(0,1),
            'south_east':(1,1),
            'south':(1,0),
            'south_west':(1,-1),
            'west':(1,-1),
            'north_west':(-1,-1)
            }

CARDINALTOCARTESIANPLOT = {
            'north':(0,-1),
            'north_east':(1,-1),
            'east':(1,0),
            'south_east':(1,1),
            'south':(0,1),
            'south_west':(-1,1),
            'west':(-1,0),
            'north_west':(-1,-1)
            }

CARTESIANTOCARDINAL = {
            (0,-1):'north',
            (1,-1):'north_east',
            (1,0):'east',
            (1,1):'south_east',
            (0,1):'south',
            (-1,1):'south_west',
            (-1,0):'west',
            (-1,-1):'north_west'
            }

SHAPEFEATURES = ('area', 'perimeter', 'circleness', 'rectangleness',
                'H', 'W', 'xTile', 'yTile', 'savePath')

DRAWING = {
    'margin': 5,
    'shapes': ['rectangle', 'circle'],
    'shapes_distribution' : [0.85, 0.15],
    'stripe_start'  : 7,
    'stripe_width'  : 3,
    'stripe_spacing': 10,
    'max_size_dict' : {'xs':45, 's':90, 'm':181, 'l':362},
    'size_to_shirt' : {64:'xs', 128:'s', 256:'m',  512:'l'},
    'kernel':362,
    'stride':20
}

SHAPEDISTRIBUTION = ['A_distribution', 'H_distribution', 'W_distribution']
