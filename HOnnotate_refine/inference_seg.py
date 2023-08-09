import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0,os.path.join(os.getcwd(), 'models'))
sys.path.insert(0,os.path.join(os.getcwd(), 'models/slim'))
from HOdatasets.commonDS import *
import models.deeplab.common as common
from eval import evalSeg
from eval import eval2DKps
from utils import inferenceUtils as infUti
from HOdatasets.mypaths import *
from utils.predictSegHandObject import getNetSess
from onlineAug.commonAug import networkData
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import multiprocessing as mlp
import warnings
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
from absl import flags
from absl import app

FLAGS = flags.FLAGS
# flags.DEFINE_string('seq', 'test', 'Sequence Name') # name ,default, help
# flags.DEFINE_string('camID', '0', 'Cam ID') # name ,default, help
# flags.DEFINE_integer('start', 0, 'Cam ID') # name ,default, help
# flags.DEFINE_integer('end', 2300, 'Cam ID') # name ,default, help
flags.DEFINE_string('db', '230612', 'target db Name') # name ,default, help
# flags.DEFINE_string('seq', 'bowl_18_00', 'Sequence Name')
# flags.DEFINE_string('camID', 'mas', 'target camera')
camIDset = ['mas', 'sub1', 'sub2', 'sub3']

dataset_mix = infUti.datasetMix.OXR_MULTICAMERA
w = 640
h = 480
numConsThreads = 1

# baseDir = HO3D_MULTI_CAMERA_DIR
baseDir = OXR_MULTI_CAMERA_DIR


def postProcess(dummy, consQueue, numImgs, numConsThreads):
    while True:

        queueElem = consQueue.get()
        predsDict = queueElem[0]
        ds = queueElem[1]
        jobID = queueElem[2]

        croppedImg = predsDict[common.IMAGE]
        if common.SEMANTIC in predsDict.keys():

            predsDict[common.SEMANTIC] = predsDict[common.SEMANTIC][0]

            assert len(ds.fileName.split('/')) == 3, 'Dont know this filename format'

            seq = ds.fileName.split('/')[0]
            camInd = ds.fileName.split('/')[1]
            id = ds.fileName.split('/')[2]

            finalSaveDir =  os.path.join(baseDir, FLAGS.db, seq, 'segmentation_deep', str(camInd), 'visualization')
            finalRawSaveDir = os.path.join(baseDir, FLAGS.db, seq, 'segmentation_deep', str(camInd), 'raw_seg_results')


            labelFullImg = np.zeros_like(ds.imgRaw)[:,:,0]
            patchSize = predsDict['bottomRight'] - predsDict['topLeft']

            scaleW = float(patchSize[0]) / float(w)
            scaleH = float(patchSize[1]) / float(h)
            labelPatch = cv2.resize(np.expand_dims(predsDict[common.SEMANTIC],2).astype(np.uint8),
                                    (int(ds.imgRaw.shape[1]*scaleW), int(ds.imgRaw.shape[0]*scaleH)),
                                    interpolation=cv2.INTER_NEAREST)
            labelFullImg[predsDict['topLeft'][1]:predsDict['bottomRight'][1], predsDict['topLeft'][0]:predsDict['bottomRight'][0]] = labelPatch

            # save predictions
            evalSeg.saveAnnotations(predsDict[common.SEMANTIC], croppedImg,
                                    finalSaveDir, id,
                                    raw_save_dir=finalRawSaveDir,
                                    also_save_raw_predictions=True, fullRawImg=labelFullImg)

        print('Frame %d of %d, (%s)' % (jobID, numImgs, ds.fileName))
        if jobID>=(numImgs-numConsThreads):
            return


def runNetInLoop(fileListIn, numImgs, camID, seq):
    myG = tf.Graph()

    with myG.as_default():
        data = networkData(image=tf.placeholder(tf.uint8, shape=(h, w, 3)),
                           label=tf.placeholder(tf.uint8, shape=(h, w, 1)),
                           kps2D=None,
                           kps3D=None,
                           imageID='0',
                           h=h,
                           w=w,
                           outType=None,
                           dsName=None,
                           camMat=None)

    sess, g, predictions, dataPreProcDict = getNetSess(data, h, w, myG)

    dsQueue, dsProcs = infUti.startInputQueueRunners(dataset_mix, splitType.TEST, FLAGS.db, seq, camID, numThreads=5, isRemoveBG=False, fileListIn=fileListIn)

    # start consumer threads
    consQueue = mlp.Queue(maxsize=100)
    procs = []
    for proc_index in range(numConsThreads):
        args = ([], consQueue, numImgs, numConsThreads)
        proc = mlp.Process(target=postProcess, args=args)
        # proc.daemon = True

        proc.start()
        procs.append(proc)

    # start the network
    for i in range(numImgs):

        while(dsQueue.empty()):
            waitTime = 10*1e-3
            time.sleep(waitTime)

        ds = dsQueue.get()

        assert isinstance(ds, dataSample)

        startTime = time.time()
        predsDict = sess.run(predictions, feed_dict={data.image: ds.imgRaw},)
        # print('Runtime = %f'%(time.time() - startTime))

        labels = predsDict[common.SEMANTIC]
        labels[labels == 1] = 1
        labels[labels == 2] = 2
        labels[labels == 3] = 2
        predsDict[common.SEMANTIC] = labels

        consQueue.put([predsDict, ds, i])

    for proc in procs:
        proc.join()

    while(not consQueue.empty()):
        time.sleep(10*1e-3)

    consQueue.close()
    dsQueue.close()




def main(argv):

    rootDir = os.path.join(baseDir, FLAGS.db)
    total_seq = len(os.listdir(rootDir))
    for seqIdx, seq in enumerate(sorted(os.listdir(rootDir))):
        
        d = os.path.join(rootDir, seq)
        if os.path.isdir(d):            
            print("(%s in %s) : %s" % (seqIdx, total_seq, seq))
            
            if not os.path.exists(os.path.join(baseDir, FLAGS.db, seq, 'segmentation_deep')):
                os.mkdir(os.path.join(baseDir, FLAGS.db, seq, 'segmentation_deep'))
                for camID in camIDset:
                    os.mkdir(os.path.join(baseDir, FLAGS.db, seq, 'segmentation_deep', camID))
                    os.mkdir(os.path.join(baseDir, FLAGS.db, seq, 'segmentation_deep', camID, 'visualization'))
                    os.mkdir(os.path.join(baseDir, FLAGS.db, seq, 'segmentation_deep', camID, 'raw_seg_results'))
            
            for camID in camIDset:
                fileListIn = os.listdir(join(baseDir, FLAGS.db, seq, 'rgb_crop', camID))
                fileListIn = [join(seq, camID, f[:-4]) for f in fileListIn if 'png' in f]
                fileListIn = sorted(fileListIn)

                numImgs = len(fileListIn)

                runNetInLoop(fileListIn, numImgs, camID, seq)

if __name__ == '__main__':
    app.run(main)







