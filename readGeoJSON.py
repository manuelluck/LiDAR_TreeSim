import os
import subprocess
from pathlib import Path
os.environ['MPLBACKEND'] = 'TkAgg'

# data handling
import geojson
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

# pdf creation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER

# Register the fonts
pdfmetrics.registerFont(TTFont('Calibri-Light', 'calibril.ttf'))
pdfmetrics.registerFont(TTFont('Times-Roman', 'times.ttf'))


# output Folders Variables
imageFolder = 'H:\\ForestSimulation\\PythonFigures\\'
pdfFolder   = 'H:\\ForestSimulation\\PythonPDF\\'

dwFolder   = 'H:\\Blender\\csv4blender\\'
treeFolder = 'H:\\Blender\\csv4blender\\'

# DataFiles:
dwPath      = 'C:\\Users\\luckmanu\\Downloads\\deadwood_data\\dw_2022.GeoJSON'
treePath    = 'C:\\Users\\luckmanu\\Downloads\\deadwood_data\\trees_2022.GeoJSON'

# PositionFile:
txtPath     = 'C:\\Users\\luckmanu\\Downloads\\deadwood_data\\possiblePlots2.txt'

# class:
class GeoJSON2SimCloud:
    def __init__(self, wdir: str = '',heliosMain: str = 'H:/Helios/helios-plusplus-win/',allowFolderCreation=False):
        self.path = dict()
        self.path['helios'] = heliosMain
        # set up working directory
        try:
            if wdir == '':
                self.path['wdir'] = Path(__file__).parent
            else:
                if Path(wdir).exists():       # This determines if the string input is a valid path
                    if Path(wdir).is_dir():
                        self.path['wdir'] = Path(wdir)
                        print(f'Working directory set:\n{self.path["wdir"]}\n')
                    elif Path(wdir).is_file():
                        self.path['wdir'] = Path(wdir).parent
                        print(f'Working directory set:\n{self.path["wdir"]}\n')
                elif allowFolderCreation:
                    os.makedirs(wdir)
                    print(f'Folder is created: {wdir}\n!')
                    self.path['wdir'] = Path(wdir)
                    print(f'Working directory set:\n{self.path["wdir"]}\n')
                else:
                    print(f'The folder ({wdir}) does not exist and "allowFolderCreation" is set on False\n')
        except:
            print('Error while setting up working directory!\n'
                  'Check all Paths for spelling mistakes or insufficient allowances\n')

        try:
            self.projectCount = max([int(folder.name.split('_')[1]) for folder in self.path['wdir'].iterdir()
                                     if (folder.name.startswith('Project') and folder.is_dir())])
        except:
            self.projectCount = 0

        self.projects = [None for _ in range(self.projectCount)]
        self.currentProject = None

    class Project:
        def __init__(self,projectNr,wdir,trees,dw,plots,main):
            self.main           = main
            self.dwData         = None
            self.treeData       = None
            self.plotPositions  = []
            self.plots          = dict()
            self.id             = projectNr

            self.projectDir = wdir.joinpath(f'Project_{projectNr:03d}')
            self.projectDir.mkdir(parents=True, exist_ok=True)

            self.path               = dict()
            self.path['wdir']       = Path(self.projectDir)
            self.path['meta']       = self.path['wdir'].joinpath('MetaData')
            self.path['export']     = self.path['wdir'].joinpath('Export')
            self.path['blender']    = self.path['wdir'].joinpath('Blender')
            self.path['helios']     = self.path['wdir'].joinpath('Helios')
            self.path['trajectory'] = self.path['helios'].joinpath('Trajectory')
            self.path['pdf']        = self.path['export'].joinpath('PDF')
            self.path['png']        = self.path['export'].joinpath('PNG')

            if Path(trees).is_file():
                if str(trees).endswith('GeoJSON'):
                    self.path['treeData']   = Path(trees)
                else:
                    print('Currently only GeoJSON for Trees and DW data!')
            else:
                print('Tree file not found!')

            if Path(dw).is_file():
                if str(dw).endswith('GeoJSON'):
                    self.path['dwData']   = Path(dw)
                else:
                    print('Currently only GeoJSON for Trees and DW data!')
            else:
                print('Deadwood file not found!')

            if Path(plots).is_file():
                if str(plots).endswith('txt'):
                    self.path['plotPositions']   = Path(plots)
                else:
                    print('Currently only txt file for plot-positions!')
            else:
                print('Plot-positions file not found!')

            self.path['meta'].mkdir(parents=True, exist_ok=True)
            self.path['export'].mkdir(parents=True, exist_ok=True)
            self.path['blender'].mkdir(parents=True, exist_ok=True)
            self.path['pdf'].mkdir(parents=True, exist_ok=True)
            self.path['png'].mkdir(parents=True, exist_ok=True)
            self.path['helios'].mkdir(parents=True, exist_ok=True)
            self.path['trajectory'].mkdir(parents=True, exist_ok=True)

            self.loadDwGeoJSON()
            self.loadTreeGeoJSON()
            self.loadPosData()

            # Define different Plot extend and Scanline:
            self.scanPatterns = dict()
            self.scanPatterns['LFI']                = dict()
            self.scanPatterns['LFI']['extend']      = [[-25,25],[25,-25]]
            self.updateScanLine()  # scanline is derived from extend -- callable update if extend is changed
            self.createLeg4HeliosMLS()  # create trj file for Scanline

        def updateScanLine(self):
            self.scanPatterns['LFI']['scanLine'] = [[0, 0],
                                                    [self.scanPatterns['LFI']['extend'][0][0],
                                                     self.scanPatterns['LFI']['extend'][1][0]],
                                                    [self.scanPatterns['LFI']['extend'][0][1],
                                                     self.scanPatterns['LFI']['extend'][1][0]],
                                                    [self.scanPatterns['LFI']['extend'][0][1],
                                                     self.scanPatterns['LFI']['extend'][1][0] / 2],
                                                    [self.scanPatterns['LFI']['extend'][0][0],
                                                     self.scanPatterns['LFI']['extend'][1][0] / 2],
                                                    [self.scanPatterns['LFI']['extend'][0][0],
                                                     0],
                                                    [self.scanPatterns['LFI']['extend'][0][1],
                                                     0],
                                                    [self.scanPatterns['LFI']['extend'][0][1],
                                                     self.scanPatterns['LFI']['extend'][1][1] / 2],
                                                    [self.scanPatterns['LFI']['extend'][0][0],
                                                     self.scanPatterns['LFI']['extend'][1][1] / 2],
                                                    [self.scanPatterns['LFI']['extend'][0][0],
                                                     self.scanPatterns['LFI']['extend'][1][1]],
                                                    [self.scanPatterns['LFI']['extend'][0][1],
                                                     self.scanPatterns['LFI']['extend'][1][1]],
                                                    [0, 0]]
        def createLeg4HeliosMLS(self,rpmRoll=26,vWalk=1.5,turnTime=0.1):
            self.scanPatterns['LFI']['legs'] = ['#TIME_COLUMN: 0',
                                                '#HEADER: "t", "pitch", "roll", "yaw", "x", "y", "z"']
            t       = 0
            roll    = 0
            yaw     = 0

            for i in range(len(self.scanPatterns['LFI']['scanLine'])):
                if i == 0:
                    # starting point on Tripod (pitch = 0)
                    pos     = self.scanPatterns['LFI']['scanLine'][i]
                    t       = 0
                    roll    = 0
                    pitch   = 105
                    yaw     = 0
                    x       = pos[0]
                    y       = pos[1]
                    z       = 1.2
                    self.scanPatterns['LFI']['legs'].append(f'{t}, {pitch}, {roll}, {yaw}, {x}, {y}, {z}')

                    # turning towards next pos:
                    posNext = self.scanPatterns['LFI']['scanLine'][i + 1]
                    dx      = posNext[0]-pos[0]
                    dy      = posNext[1]-pos[1]
                    if dy == 0 and dx < 0:
                        yaw = 270
                    elif dy == 0 and dx > 0:
                        yaw = 90
                    else:
                        yaw = np.degrees(np.arctan(dx/dy))
                        if yaw < 0:
                            yaw = 360 + yaw
                    t       += turnTime
                    self.scanPatterns['LFI']['legs'].append(f'{t}, {pitch}, {roll}, {yaw}, {x}, {y}, {z}')

                elif i == len(self.scanPatterns['LFI']['scanLine'])-1:
                    # arriving at end pos:
                    pos     = self.scanPatterns['LFI']['scanLine'][i]
                    posOld  = self.scanPatterns['LFI']['scanLine'][i-1]
                    dx      = posOld[0]-pos[0]
                    dy      = posOld[1]-pos[1]
                    l       = (dx**2 + dy**2)**(1/2)
                    dt      = l/vWalk
                    dRoll   = (360*(rpmRoll/60))*dt
                    t       += dt
                    roll    += dRoll
                    pitch   = 90
                    yaw     += 0
                    x       = pos[0]
                    y       = pos[1]
                    z       = 1.2
                    self.scanPatterns['LFI']['legs'].append(f'{t}, {pitch}, {roll}, {yaw}, {x}, {y}, {z}')

                else:
                    # arriving at intermediate pos:
                    pos     = self.scanPatterns['LFI']['scanLine'][i]
                    posOld  = self.scanPatterns['LFI']['scanLine'][i-1]
                    dx      = posOld[0]-pos[0]
                    dy      = posOld[1]-pos[1]
                    l       = (dx**2 + dy**2)**(1/2)
                    dt      = l/vWalk
                    dRoll   = (360*(rpmRoll/60))*dt
                    t       += dt
                    roll    += dRoll
                    pitch   = 90
                    yaw     += 0
                    x       = pos[0]
                    y       = pos[1]
                    z       = 1.2
                    self.scanPatterns['LFI']['legs'].append(f'{t}, {pitch}, {roll}, {yaw}, {x}, {y}, {z}')

                    # turning towards next pos:
                    posNext = self.scanPatterns['LFI']['scanLine'][i + 1]
                    dx      = posNext[0]-pos[0]
                    dy      = posNext[1]-pos[1]
                    if dy == 0 and dx < 0:
                        yaw = 270
                    elif dy == 0 and dx > 0:
                        yaw = 90
                    else:
                        yaw = np.degrees(np.arctan(dx/dy))
                        if yaw < 0:
                            yaw = 360 + yaw
                    t       += turnTime
                    self.scanPatterns['LFI']['legs'].append(f'{t}, {pitch}, {roll}, {yaw}, {x}, {y}, {z}')

            with open(self.path['trajectory'].joinpath('MLS.trj'), mode="wt") as f:
                for line in self.scanPatterns['LFI']['legs']:
                    f.write(f'{line}\n')

        def writeXml4Helios(self,trj='MLS.trj',scannerPath='data/scanners_tls.xml',
                            scannerName='vlp16',headRotSpeed = 3600,pulseFreq=180000):
            for plot in self.plots.keys():
                groundObj   = str(self.plots[plot]['ground4Helios'])
                grassObj    = str(self.plots[plot]['grass4Helios'])
                lowVegObj   = str(self.plots[plot]['lowVeg4Helios'])
                dwObj       = str(self.plots[plot]['dw4Helios'])
                treeObj     = str(self.plots[plot]['tree4Helios'])

                self.plots[plot]['scannerPath']     = scannerPath
                self.plots[plot]['scanner']         = scannerName
                self.plots[plot]['scenePath']       = self.plots[plot]['heliosPath'].joinpath(f'xml/scene_{plot}.xml')
                self.plots[plot]['sceneName']       = f'scene_{plot}'
                self.plots[plot]['surveyPath']      = self.plots[plot]['heliosPath'].joinpath(
                                                                        f'xml/{self.plots[plot]["plotName"]}.xml')
                self.plots[plot]['xmlFolder']       = self.plots[plot]['heliosPath'].joinpath('xml/')
                self.plots[plot]['xmlFolder'].mkdir(parents=True,exist_ok=True)

                surveyXML  = [f'<?xml version="1.0" encoding="UTF-8"?>',
                              f'<document>',
                              f'\t<scannerSettings id="scaset" active="true" pulseFreq_hz="{pulseFreq}" scanFreq_hz="100"/>',
                              f'\t<survey name="{self.plots[plot]["plotName"]}" ',
                              f'\t\tscene="{self.plots[plot]["scenePath"]}#{self.plots[plot]["sceneName"]}" ',
                              f'\t\tplatform="interpolated" basePlatform="data/platforms.xml#sr22" ',
                              f'\t\tscanner="{self.plots[plot]["scannerPath"]}#{self.plots[plot]["scanner"]}">'
                              ]

                for l in range(3,len(self.scanPatterns["LFI"]["legs"])):
                    if l > 0:
                        if l == 3:
                            headRotStart = float(self.scanPatterns["LFI"]["legs"][l-1].split(",")[0])*headRotSpeed
                            headRotStop  = float(self.scanPatterns["LFI"]["legs"][l].split(",")[0])*headRotSpeed
                            surveyXML += [f'\t\t<leg>',
                                          f'\t\t\t<platformSettings ',
                                          f'\t\t\t\ttrajectory="{self.path["trajectory"].joinpath(trj)}"',
                                          f'\t\t\t\ttIndex="0" xIndex="4" yIndex="5" zIndex="6" ',
                                          f'\t\t\t\trollIndex="2" pitchIndex="1" yawIndex="3"',
                                          f'\t\t\t\tslopeFilterThreshold="0.0" toRadians="true" syncGPSTime="true"',
                                          f'\t\t\t\ttStart="{self.scanPatterns["LFI"]["legs"][l-1].split(",")[0]}"',
                                          f'\t\t\t\ttEnd="{self.scanPatterns["LFI"]["legs"][l].split(",")[0]}"',
                                          f'\t\t\t\tteleportToStart="true"',
                                          f'\t\t\t/>',
                                          f'\t\t\t<scannerSettings template="scaset"',
                                          f'\t\t\t\theadRotatePerSec_deg="{headRotSpeed}"',
                                          f'\t\t\t\theadRotateStart_deg="{headRotStart}" ',
                                          f'\t\t\t\theadRotateStop_deg="{headRotStop}"',
                                          f'\t\t\t\ttrajectoryTimeInterval_s="0.01"/>',
                                          f'\t\t</leg>'
                                          ]
                        else:
                            headRotStart = float(self.scanPatterns["LFI"]["legs"][l-1].split(",")[0])*headRotSpeed
                            headRotStop  = float(self.scanPatterns["LFI"]["legs"][l].split(",")[0])*headRotSpeed
                            surveyXML += [f'\t\t<leg>',
                                          f'\t\t\t<platformSettings ',
                                          f'\t\t\t\ttrajectory="{self.path["trajectory"].joinpath(trj)}"',
                                          f'\t\t\t\ttStart="{self.scanPatterns["LFI"]["legs"][l-1].split(",")[0]}"',
                                          f'\t\t\t\ttEnd="{self.scanPatterns["LFI"]["legs"][l].split(",")[0]}"',
                                          f'\t\t\t\tteleportToStart="true"',
                                          f'\t\t\t/>',
                                          f'\t\t\t<scannerSettings template="scaset"',
                                          f'\t\t\t\theadRotatePerSec_deg="{headRotSpeed}"',
                                          f'\t\t\t\theadRotateStart_deg="{headRotStart}" ',
                                          f'\t\t\t\theadRotateStop_deg="{headRotStop}"',
                                          f'\t\t\t\ttrajectoryTimeInterval_s="0.01"/>',
                                          f'\t\t</leg>'
                                          ]

                surveyXML += [f'\t</survey>',
                              f'</document>']

                sceneXML = [f'<?xml version="1.0" encoding="UTF-8"?>',
                            f'<document>',
                            f'\t<scene id="{self.plots[plot]["sceneName"]}" name="{self.plots[plot]["sceneName"]}">',
                            f'\t\t<part>',
                            f'\t\t\t<filter type="objloader">',
                            f'\t\t\t\t<param type="string" key="filepath" value="{str(self.plots[plot]["ground4Helios"])}" />',
                            f'\t\t\t</filter>',
                            f'\t\t</part>',
                            f'\t\t<part>',
                            f'\t\t\t<filter type="objloader">',
                            f'\t\t\t\t<param type="string" key="filepath" value="{str(self.plots[plot]["grass4Helios"])}" />',
                            f'\t\t\t</filter>',
                            f'\t\t</part>',
                            f'\t\t<part>',
                            f'\t\t\t<filter type="objloader">',
                            f'\t\t\t\t<param type="string" key="filepath" value="{str(self.plots[plot]["lowVeg4Helios"])}" />',
                            f'\t\t\t</filter>',
                            f'\t\t</part>',
                            f'\t\t<part>',
                            f'\t\t\t<filter type="objloader">',
                            f'\t\t\t\t<param type="string" key="filepath" value="{str(self.plots[plot]["tree4Helios"])}" />',
                            f'\t\t\t</filter>',
                            f'\t\t</part>',
                            f'\t\t<part>',
                            f'\t\t\t<filter type="objloader">',
                            f'\t\t\t\t<param type="string" key="filepath" value="{str(self.plots[plot]["dw4Helios"])}" />',
                            f'\t\t\t</filter>',
                            f'\t\t</part>',
                            f'\t</scene>',
                            f'</document>'
                            ]
                with open(self.plots[plot]['surveyPath'], mode="wt") as t:
                    for line in surveyXML:
                        t.write(f'{line}\n')
                with open(self.plots[plot]['scenePath'], mode="wt") as t:
                    for line in sceneXML:
                        t.write(f'{line}\n')

        def runHelios(self,flags=''):
            for plot in self.plots.keys():

                subprocess.run([f'{self.main.path["helios"]}run/helios.exe',
                                f'{str(self.plots[plot]["surveyPath"])}',
                                f'{flags}'],
                               cwd=self.main.path["helios"])

        def combineHeliosLegs(self,removeLegs=True):
            for plot in self.plots.keys():
                self.plots[plot]['heliosOutputDir'] = self.plots[plot]['heliosPath'].joinpath('output/')
                self.plots[plot]['heliosOutputDir'].mkdir(parents=True,exist_ok=True)

                folders = [os.path.join(f'{self.main.path["helios"]}output', entry) for entry in os.listdir(f'{self.main.path["helios"]}output')
                           if os.path.isdir(os.path.join(f'{self.main.path["helios"]}output', entry))]
                for folder in folders:
                    if plot in folder:
                        subFolders = [os.path.join(f'{folder}', entry) for entry in os.listdir(f'{folder}')
                                      if os.path.isdir(os.path.join(f'{folder}', entry))]
                        for subFolder in subFolders:
                            #trajFiles   = [file for file in os.listdir(subFolder) if file.endswith('.txt')]
                            xyzFiles    = [file for file in os.listdir(subFolder) if file.endswith('.xyz')]

                            for xyzFile in xyzFiles:

                                with open(xyzFile,'r') as f:
                                    with open(f'{self.plots[plot]["heliosOutputDir"].joinpath(self.plots[plot]["sceneName"])}.xyz' , "a") as w:
                                        for line in f:
                                            x,y,z,_,_,_,_,_,label,_,_ = line.split()
                                            w.write(f"{x} {y} {z} {label}\n")
                            os.rmdir(subFolder)

        def loadDwGeoJSON(self):
            with open(self.path['dwData'], 'r') as f:
                gj = geojson.load(f)
                self.dwData = gj['features']

        def loadTreeGeoJSON(self):
            with open(self.path['treeData'], 'r') as f:
                gj = geojson.load(f)
                self.treeData = gj['features']

        def loadPosData(self):
            with open(self.path['plotPositions'], 'r') as t:
                for line in t.readlines():
                    if len(line.split(',')) == 2:
                        e, n = line.split(',')
                        self.plotPositions.append([float(e),float(n)])

        def writeMetaFile(self):
            with open(self.path['meta'].joinpath('path.txt'), 'w') as f:
                for key, value in self.path.items():
                    f.write(f'{key},{value}\n')

        def plotting(self, dw, trees, scanline, figPath='figure.png'):
            """
            function - plotting(dw,trees)
            dw:     geoJSON deadwood / line-file
            trees:  geoJSon trees   / point-file
            """
            scanline = np.vstack(scanline)
            cmap = cm.get_cmap('terrain')
            plt.figure()
            for i in range(len(trees[trees.keys()[0]])):
                try:
                    t = trees.iloc[i]
                    plt.scatter(t['X'], t['Y'],
                                s=(t['D1'] / max(trees['D1']) * 20),
                                marker='o',
                                color=cmap((t['BA'] - 100) / 1000))
                except:
                    print(trees.iloc[i])
            for i in range(len(dw[dw.keys()[0]])):
                d = dw.iloc[i]
                plt.plot([d['X0'], d['X1']], [d['Y0'], d['Y1']],
                         linewidth=(d['D1'] / max(dw['D1']) * 2),
                         color=cmap((d['BA'] - 100) / 1000))

            plt.plot(scanline[:, 0], scanline[:, 1], marker='.', linestyle='--', color='tan', alpha=0.5)
            for p in [0, 1, 2, 9, 10]:
                plt.text(scanline[p, 0], scanline[p, 1] + 2,
                         f'{scanline[p, 0]}E / {scanline[p, 1]}N',
                         fontdict={'size': 6},
                         horizontalalignment='center')
            plt.axis('off')
            plt.savefig(figPath, dpi=200)

        def preparePlots(self,scanPattern='LFI',singlePlot=None):
            if singlePlot is None:
                for plot in range(len(self.plotPositions)):
                    E = self.plotPositions[plot][0]
                    N = self.plotPositions[plot][1]
                    # getting Corners
                    ul      = [E+self.scanPatterns[scanPattern]['extend'][0][0],
                               N+self.scanPatterns[scanPattern]['extend'][1][0]]
                    lr      = [E+self.scanPatterns[scanPattern]['extend'][0][1],
                               N+self.scanPatterns[scanPattern]['extend'][1][1]]

                    # selecting Features in Plot
                    dwPlot      = [dw for dw in self.dwData if any([all([dw['geometry']['coordinates'][0][0] >= ul[0],
                                                                         dw['geometry']['coordinates'][0][0] <= lr[0],
                                                                         dw['geometry']['coordinates'][0][1] <= ul[1],
                                                                         dw['geometry']['coordinates'][0][1] >= lr[1]]),
                                                                    all([dw['geometry']['coordinates'][0][0] >= ul[0],
                                                                         dw['geometry']['coordinates'][0][0] <= lr[0],
                                                                         dw['geometry']['coordinates'][0][1] <= ul[1],
                                                                         dw['geometry']['coordinates'][0][1] >= lr[1]])
                                                                    ])]
                    treesPlot   = [tree for tree in self.treeData if all([tree['geometry']['coordinates'][0] >= ul[0],
                                                                          tree['geometry']['coordinates'][0] <= lr[0],
                                                                          tree['geometry']['coordinates'][1] <= ul[1],
                                                                          tree['geometry']['coordinates'][1] >= lr[1]])]

                    plot4blend = self.path['blender'].joinpath(f'plot_{plot:03d}')
                    plot4blend.mkdir(parents=True, exist_ok=True)

                    dfTree  = self.convert2pd(treesPlot,[E,N])
                    dfTree  = self.cleanDf(dfTree)
                    dfTree.to_csv(plot4blend.joinpath('tree.csv'))

                    dfDw    = self.convert2pd(dwPlot,[E,N])
                    dfDw    = self.cleanDf(dfDw)
                    dfDw.to_csv(plot4blend.joinpath('dw.csv'))

                    self.plots[f'{plot:03d}']               = dict()
                    self.plots[f'{plot:03d}']['heliosPath'] = self.path['helios'].joinpath(f'plot_{plot:03d}')
                    self.plots[f'{plot:03d}']['blenderPath']= self.path['blender'].joinpath(f'plot_{plot:03d}')
                    self.plots[f'{plot:03d}']['plotName']   = f'Plot_{plot:03d}'
                    self.plots[f'{plot:03d}']['dwPath']     = plot4blend.joinpath('dw.csv')
                    self.plots[f'{plot:03d}']['treePath']   = plot4blend.joinpath('tree.csv')
                    self.plots[f'{plot:03d}']['dwData']     = dfDw
                    self.plots[f'{plot:03d}']['treeData']   = dfTree
                    self.plots[f'{plot:03d}']['center']     = self.plotPositions[plot]
                    self.plots[f'{plot:03d}']['scanLine']   = np.vstack([[s[0]+E,s[1]+N] for s in
                                                                         self.scanPatterns[scanPattern]['scanLine']])

                    self.plots[f'{plot:03d}']['blenderPath'].mkdir(parents=True, exist_ok=True)
                    self.plots[f'{plot:03d}']['heliosPath'].mkdir(parents=True,exist_ok=True)
                    self.plotting(dfDw, dfTree, self.plots[f'{plot:03d}']['scanLine'],
                                  figPath=self.path['png'].joinpath(f'Plot_{plot:03d}.png'))
            else:
                plot = int(singlePlot)
                E = self.plotPositions[plot][0]
                N = self.plotPositions[plot][1]
                # getting Corners
                ul = [E + self.scanPatterns[scanPattern]['extend'][0][0],
                      N + self.scanPatterns[scanPattern]['extend'][1][0]]
                lr = [E + self.scanPatterns[scanPattern]['extend'][0][1],
                      N + self.scanPatterns[scanPattern]['extend'][1][1]]

                # selecting Features in Plot
                dwPlot = [dw for dw in self.dwData if any([all([dw['geometry']['coordinates'][0][0] >= ul[0],
                                                                dw['geometry']['coordinates'][0][0] <= lr[0],
                                                                dw['geometry']['coordinates'][0][1] <= ul[1],
                                                                dw['geometry']['coordinates'][0][1] >= lr[1]]),
                                                           all([dw['geometry']['coordinates'][0][0] >= ul[0],
                                                                dw['geometry']['coordinates'][0][0] <= lr[0],
                                                                dw['geometry']['coordinates'][0][1] <= ul[1],
                                                                dw['geometry']['coordinates'][0][1] >= lr[1]])
                                                           ])]
                treesPlot = [tree for tree in self.treeData if all([tree['geometry']['coordinates'][0] >= ul[0],
                                                                    tree['geometry']['coordinates'][0] <= lr[0],
                                                                    tree['geometry']['coordinates'][1] <= ul[1],
                                                                    tree['geometry']['coordinates'][1] >= lr[1]])]

                plot4blend = self.path['blender'].joinpath(f'plot_{plot:03d}')
                plot4blend.mkdir(parents=True, exist_ok=True)

                dfTree = self.convert2pd(treesPlot, [E, N])
                dfTree = self.cleanDf(dfTree)
                dfTree.to_csv(plot4blend.joinpath('tree.csv'))

                dfDw = self.convert2pd(dwPlot, [E, N])
                dfDw = self.cleanDf(dfDw)
                dfDw.to_csv(plot4blend.joinpath('dw.csv'))

                self.plots[f'{plot:03d}'] = dict()
                self.plots[f'{plot:03d}']['heliosPath'] = self.path['helios'].joinpath(f'plot_{plot:03d}')
                self.plots[f'{plot:03d}']['blenderPath']= self.path['blender'].joinpath(f'plot_{plot:03d}')

                self.plots[f'{plot:03d}']['plotName']   = f'Plot_{plot:03d}'
                self.plots[f'{plot:03d}']['dwPath']     = plot4blend.joinpath('dw.csv')
                self.plots[f'{plot:03d}']['treePath']   = plot4blend.joinpath('tree.csv')
                self.plots[f'{plot:03d}']['dwData']     = dfDw
                self.plots[f'{plot:03d}']['treeData']   = dfTree
                self.plots[f'{plot:03d}']['center']     = self.plotPositions[plot]
                self.plots[f'{plot:03d}']['scanLine']   = np.vstack([[s[0] + E, s[1] + N] for s in
                                                                     self.scanPatterns[scanPattern]['scanLine']])

                self.plots[f'{plot:03d}']['blenderPath'].mkdir(parents=True, exist_ok=True)
                self.plots[f'{plot:03d}']['heliosPath'].mkdir(parents=True, exist_ok=True)

                self.plotting(dfDw, dfTree, self.plots[f'{plot:03d}']['scanLine'],
                              figPath=self.path['png'].joinpath(f'Plot_{plot:03d}.png'))


        def convert2pd(self,geoJSONdat,offset):
            pointCollector  = []
            lineCollector   = []
            for element in geoJSONdat:
                if element['geometry']['type'] == 'Point':
                    try:
                        pointCollector.append(np.hstack([element['geometry']['coordinates'],
                                                         element['properties']['d1'],
                                                         element['properties']['ba'],offset]))
                    except:
                        print(element['properties'])

                elif element['geometry']['type'] == 'LineString':
                    try:
                        p1  = element['geometry']['coordinates'][0]
                        p2  = element['geometry']['coordinates'][1]
                        m = [p1[0]+(p2[0]-p1[0])/2,p1[1]+(p2[1]-p1[1])/2]
                        l   = ((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**(1/2)
                        if p1[0]-p2[0] == 0:
                            if p1[1]-p2[1] > 0:
                                a = 90
                            else:
                                a = 270
                        else:
                            a   = np.degrees(np.tan((p1[1]-p2[1])/p1[0]-p2[0]))
                        m = p1
                        lineCollector.append(np.hstack([element['geometry']['coordinates'][0],
                                                        element['geometry']['coordinates'][1],
                                                        element['properties']['d1_start'],
                                                        element['properties']['species'],
                                                        l,
                                                        a,
                                                        m[0],
                                                        m[1],
                                                        offset]))
                    except:
                        print(element['properties'])

            if len(pointCollector) > 0 and len(lineCollector) > 0:
                pointDf = pd.DataFrame(np.vstack(pointCollector), columns=['X', 'Y','D1','BA','Xo','Yo'])
                lineDf  = pd.DataFrame(np.vstack(lineCollector), columns=['X0', 'Y0', 'X1', 'Y1', 'D1', 'BA',
                                                                          'L', 'A','Xc','Yc','Xo','Yo'])
                return [pointDf, lineDf]

            elif len(pointCollector) > 0:
                pointDf = pd.DataFrame(np.vstack(pointCollector), columns=['X', 'Y', 'D1', 'BA','Xo','Yo'])
                return pointDf

            elif len(lineCollector) > 0:
                lineDf = pd.DataFrame(np.vstack(lineCollector), columns=['X0', 'Y0', 'X1', 'Y1', 'D1', 'BA',
                                                                         'L', 'A','Xc','Yc','Xo','Yo'])
                return lineDf

        def cleanDf(self,df):
            idx2drop = []
            for i in range(len(df[df.keys()[0]])):
                if any(np.array(df.loc[i]) == None):
                    idx2drop.append(i)

            return df.drop(idx2drop)

        def blenderRunPlots(self,blenderExePath='C:\\Users\\luckmanu\\Tools\\Blender\\blender.exe',
                                 blenderScriptPath='H:\\Blender\\Scripts\\plantFromCSV.py',
                                 groundPath='H:\\Blender\\buildingBlocks\\ground.obj',
                                 dwPath='H:\\Blender\\buildingBlocks\\layingCylinder.obj',
                                 treePath='H:\\Blender\\buildingBlocks\\standingCylinder.obj',
                                 grassPath='H:\\Blender\\Grass\\grass.obj',
                                 lowVegPath='H:\\Blender\\LowVeg\\lowVeg.004.obj'
                            ):

            for plot in self.plots.keys():
                print(f'{self.plots[plot]["plotName"]}:\n--------------\nCreating DeadWood- and Trees-Objects with Blender\n')
                self.plots[plot]['ground4Helios']   = Path(groundPath)
                self.plots[plot]['grass4Helios']    = Path(grassPath)
                self.plots[plot]['lowVeg4Helios']   = Path(lowVegPath)
                self.plots[plot]['dw4Helios']       = Path(str(self.plots[plot]['dwPath'])[:-4]+'.obj')
                self.plots[plot]['tree4Helios']     = Path(str(self.plots[plot]['treePath'])[:-4] + '.obj')
                subprocess.run([f'{blenderExePath}',f'-b',f'--python',f'{blenderScriptPath}',f'{groundPath}',
                                f'{dwPath}',f'{treePath}',f'{str(self.plots[plot]["dwPath"])}',
                                f'{str(self.plots[plot]["treePath"])}',f'{str(self.plots[plot]["dw4Helios"])}',
                                f'{str(self.plots[plot]["tree4Helios"])}'],
                               shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


    def newProject(self,treePath,dwPath,plotPath):
        if self.currentProject is not None:
            self.projects.append(self.currentProject)
        self.projectCount += 1
        self.currentProject = self.Project(projectNr=self.projectCount,
                                           wdir=self.path['wdir'],
                                           trees=treePath,
                                           dw=dwPath,
                                           plots=plotPath,
                                           main=self
                                          )

    def loadProject(self,metaFilePath):
        pass


test = GeoJSON2SimCloud(wdir='D:\\Simulation')

test.newProject(treePath=treePath,dwPath=dwPath,plotPath=txtPath)
test.currentProject.preparePlots()
test.currentProject.blenderRunPlots()
test.currentProject.writeXml4Helios(pulseFreq=18000)
test.currentProject.runHelios()
test.currentProject.writeMetaFile()






# class MetaData:
#     def __init__(self,idx=''):
#         self.path       = dict()
#         self.idx        = idx
#         self.plotPar    = dict()
#
#
#
#     def addPath(self,name,path):
#         self.path[name] = path
#
#     def create_pdf(self):
#         doc = SimpleDocTemplate(self.path['pdfFile'], pagesize=A4)
#         story = []
#         styles = getSampleStyleSheet()
#
#         titStyle = ParagraphStyle(name='Heading2',
#                                   fontName='Times-Roman',
#                                   fontSize=16,
#                                   alignment=TA_CENTER,
#                                   spaceAfter=6)
#
#         story.append(Paragraph(f'Plot: {self.idx}', titStyle))
#
#         i = Image(self.path['pltFile'])
#         img_width, img_height = i.wrap(doc.width, doc.height)
#         aspect = img_height / float(img_width)
#         i.drawHeight = aspect * doc.width
#         i.drawWidth = doc.width
#         story.append(i)
#
#         parStyle = ParagraphStyle(name='Normal', fontName='Courier',fontSize=8)
#         for par in self.path.keys():
#             spaces = (20 - len(par))*u'\xa0'
#             story.append(Paragraph(f'{par}:{spaces}{self.path[par]}', parStyle))
#         for par in self.plotPar.keys():
#             spaces = (20 - len(par))*u'\xa0'
#             story.append(Paragraph(f'{par}:{spaces}{self.plotPar[par]}', parStyle))
#
#         doc.build(story)
#
#
# # functions:
# def convert2pd(geoJSONdat,offset):
#     pointCollector  = []
#     lineCollector   = []
#     for element in geoJSONdat:
#         if element['geometry']['type'] == 'Point':
#             try:
#                 pointCollector.append(np.hstack([element['geometry']['coordinates'],
#                                                  element['properties']['d1'],
#                                                  element['properties']['ba'],offset]))
#             except:
#                 print(element['properties'])
#
#         elif element['geometry']['type'] == 'LineString':
#             try:
#                 p1  = element['geometry']['coordinates'][0]
#                 p2  = element['geometry']['coordinates'][1]
#                 m = [p1[0]+(p2[0]-p1[0])/2,p1[1]+(p2[1]-p1[1])/2]
#                 l   = ((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**(1/2)
#                 if p1[0]-p2[0] == 0:
#                     if p1[1]-p2[1] > 0:
#                         a = 90
#                     else:
#                         a = 270
#                 else:
#                     a   = np.degrees(np.tan((p1[1]-p2[1])/p1[0]-p2[0]))
#                 m = p1
#                 lineCollector.append(np.hstack([element['geometry']['coordinates'][0],
#                                                 element['geometry']['coordinates'][1],
#                                                 element['properties']['d1_start'],
#                                                 element['properties']['species'],
#                                                 l,
#                                                 a,
#                                                 m[0],
#                                                 m[1],
#                                                 offset]))
#             except:
#                 print(element['properties'])
#
#     if len(pointCollector) > 0 and len(lineCollector) > 0:
#         pointDf = pd.DataFrame(np.vstack(pointCollector), columns=['X', 'Y','D1','BA','Xo','Yo'])
#         lineDf  = pd.DataFrame(np.vstack(lineCollector), columns=['X0', 'Y0', 'X1', 'Y1', 'D1', 'BA',
#                                                                   'L', 'A','Xc','Yc','Xo','Yo'])
#         return [pointDf, lineDf]
#
#     elif len(pointCollector) > 0:
#         pointDf = pd.DataFrame(np.vstack(pointCollector), columns=['X', 'Y', 'D1', 'BA','Xo','Yo'])
#         return pointDf
#
#     elif len(lineCollector) > 0:
#         lineDf = pd.DataFrame(np.vstack(lineCollector), columns=['X0', 'Y0', 'X1', 'Y1', 'D1', 'BA',
#                                                                  'L', 'A','Xc','Yc','Xo','Yo'])
#         return lineDf
#
#
# def cleanDf(df):
#     idx2drop = []
#     for i in range(len(df[df.keys()[0]])):
#         if any(np.array(df.loc[i]) == None):
#             idx2drop.append(i)
#
#     return df.drop(idx2drop)
#
#
# def plotting(dw,trees,scanline,figPath='figure.png'):
#     """
#     function - plotting(dw,trees)
#     dw:     geoJSON deadwood / line-file
#     trees:  geoJSon trees   / point-file
#     """
#     cmap = cm.get_cmap('terrain')
#     plt.figure()
#     for i in range(len(trees[trees.keys()[0]])):
#         try:
#             t = trees.iloc[i]
#             plt.scatter(t['X'],t['Y'],
#                         s=(t['D1']/max(trees['D1'])*20),
#                         marker='o',
#                         color=cmap((t['BA']-100)/1000))
#         except:
#             print(trees.iloc[i])
#     for i in range(len(dw[dw.keys()[0]])):
#         d = dw.iloc[i]
#         plt.plot([d['X0'],d['X1']],[d['Y0'],d['Y1']],
#                  linewidth=(d['D1']/max(dw['D1'])*2),
#                  color=cmap((d['BA']-100)/1000))
#
#     plt.plot(scanline[:,0],scanline[:,1],marker='.',linestyle='--',color='tan',alpha=0.5)
#     for p in [0,1,2,9,10]:
#         plt.text(scanline[p,0],scanline[p,1]+2,
#                  f'{scanline[p,0]}E / {scanline[p,1]}N',
#                  fontdict={'size':6},
#                  horizontalalignment='center')
#     plt.axis('off')
#     plt.savefig(figPath,dpi=200)
#
#
# def create_pdf(title, fileName, imagePath, text):
#     doc = SimpleDocTemplate(fileName, pagesize=A4)
#     styles = getSampleStyleSheet()
#     t = Paragraph(title, styles["Heading2"])
#     t.style.fontName = 'Times-Roman'
#     t.style.alignment = TA_CENTER
#     i = Image(imagePath)
#     img_width, img_height = i.wrap(doc.width, doc.height)
#     aspect = img_height / float(img_width)
#     i.drawHeight = aspect * doc.width
#     i.drawWidth = doc.width
#     story = [t,i]
#     if type(text) == list:
#         for tx in text:
#             p = Paragraph(tx, styles["Normal"])
#             p.style.fontName = 'Times-Roman'
#             p.style.fontSize = 10
#             story.append(p)
#     else:
#         p = Paragraph(text, styles["Normal"])
#         p.style.fontName = 'Times-Roman'
#         p.style.fontSize = 10
#         story.append(p)
#     doc.build(story)
#
#
# # selecting Plot Origin (e.g., in QGIS)
# # E = 676473.53
# # N = 268181.67
#
# # selecting Plot Extend
# extendE = [-25,25]
# extendN = [-25,25]
#
# # defining scanline
# scanLine = [[0, 0],
#             [-25.0, 25.0],
#             [25.0, 25.0],
#             [25.0, 12.5],
#             [-25.0, 12.5],
#             [-25.0, 0.0],
#             [25.0, 0.0],
#             [25.0, -12.5],
#             [-25.0, -12.5],
#             [-25.0, -25.0],
#             [25.0, -25.0],
#             [0, 0]]
#
# # loop through all Plot Positions
# for plot in range(len(Easting)):
#     metaData = MetaData(idx=f'{plot:03d}')
#     metaData.addPath('pltFile',f'{imageFolder}Figure_{plot:03d}.png')
#     metaData.addPath('pdfFile', f'{pdfFolder}Plot_{plot:03d}.pdf')
#     metaData.addPath('dwFile', dwPath)
#     metaData.addPath('treeFile', treePath)
#     metaData.addPath('txtFile',txtPath)
#
#
#     E = Easting[plot]
#     N = Northing[plot]
#
#     # getting Corners
#     ul      = [E+extendE[0],N+extendN[1]]
#     lr      = [E+extendE[1],N+extendN[0]]
#
#     # selecting Features in Plot
#     dwPlot      = [dw for dw in dws if any([all([dw['geometry']['coordinates'][0][0] >= ul[0],
#                                                  dw['geometry']['coordinates'][0][0] <= lr[0],
#                                                  dw['geometry']['coordinates'][0][1] <= ul[1],
#                                                  dw['geometry']['coordinates'][0][1] >= lr[1]]),
#                                             all([dw['geometry']['coordinates'][0][0] >= ul[0],
#                                                  dw['geometry']['coordinates'][0][0] <= lr[0],
#                                                  dw['geometry']['coordinates'][0][1] <= ul[1],
#                                                  dw['geometry']['coordinates'][0][1] >= lr[1]])
#                                             ])]
#     treesPlot   = [tree for tree in trees if all([tree['geometry']['coordinates'][0] >= ul[0],
#                                                   tree['geometry']['coordinates'][0] <= lr[0],
#                                                   tree['geometry']['coordinates'][1] <= ul[1],
#                                                   tree['geometry']['coordinates'][1] >= lr[1]])]
#
#     dfTree = convert2pd(treesPlot,[E,N])
#     dfTree = cleanDf(dfTree)
#
#     dfDw    = convert2pd(dwPlot,[E,N])
#     dfDw    = cleanDf(dfDw)
#
#     plotScanLine = np.vstack([[s[0]+E,s[1]+N] for s in scanLine])
#     plotting(dfDw,dfTree,plotScanLine,figPath=metaData.path['pltFile'])
#     Path(f'{dwFolder}Plot_{plot:03d}').mkdir(parents=True, exist_ok=True)
#     dfDw.to_csv(f'{dwFolder}Plot_{plot:03d}\\dw.csv')
#     dfTree.to_csv(f'{dwFolder}Plot_{plot:03d}\\trees.csv')
#
#     metaData.create_pdf()
