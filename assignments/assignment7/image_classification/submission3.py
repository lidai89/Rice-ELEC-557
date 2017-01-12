import collections
import numpy as np

############################################################
# Problem 3.1

def runKMeans(k,patches,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      patches - 2D numpy array of size patchSize x numPatches
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size patchSize x k
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    centroids = np.random.randn(patches.shape[0],k)

    
    trans_patch = np.array(patches)
    numPatches=patches.shape[1]
    patchClusterLabelsV2 = np.zeros(numPatches)
    
    for i in range(maxIter):
        numCounterInCluster = np.zeros(k)       # initialize the label counter in every iteration
        #print numCounterInCluster

        # find the cluster that each patch belongs to
        for col in range(numPatches):
            _tempDiff = centroids - np.array([trans_patch[:,col],]*k).transpose()
            _norm = np.sum(_tempDiff*_tempDiff , axis=0)
            _cluster = np.argmin(_norm) # return index          #comment: could merge them together (but error is different)
            
            patchClusterLabelsV2[col] = _cluster
        #print patchClusterLabelsV2
        
        # update centroids
        centroids = np.zeros((trans_patch.shape[0],k))
        for _col in range(numPatches):
            _clusterLabel = patchClusterLabelsV2[_col]
            centroids[:, _clusterLabel:_clusterLabel+1] += trans_patch[:,_col:_col+1]
            numCounterInCluster[_clusterLabel] +=1
        centroids= centroids/numCounterInCluster
        # BEGIN_YOUR_CODE (around 19 lines of code expected)
       # raise "Not yet implemented"
        # END_YOUR_CODE

    return centroids

############################################################
# Problem 3.2

def extractFeatures(patches,centroids):
    """
    Given patches for an image and a set of centroids, extracts and return
    the features for that image.
    
    Args:
      patches - 2D numpy array of size patchSize x numPatches
      centroids - 2D numpy array of size patchSize x k
      
    Returns:
      features - 2D numpy array with new feature values for each patch
                 of the image in rows, size is numPatches x k
    """
    k = centroids.shape[1]
    numPatches = patches.shape[1]
    features = np.empty((numPatches,k))

    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    for patch in range(numPatches):
        diff=centroids-np.array([patches[:,patch],]*k).transpose()
        total=np.sum(diff*diff,axis=0)
        total=np.sqrt(total)
        totalsum=np.sum(total)
        print total
        print np.array([totalsum/k,]*k).transpose()
        features[patch,:]=np.array([totalsum/k,]*k).transpose()-total
        features=(features+abs(features))/2
    #raise "Not yet implemented"
    # END_YOUR_CODE
    return features

############################################################
# Problem 3.3.1

import math
def logisticGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of the logistic loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of logistic loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    numerator = -(2*y-1)*math.exp(-1*np.sum(featureVector*theta)*(2*y-1))
    denominator = 1 + math.exp(-1*np.sum(featureVector*theta)*(2*y-1))
    return featureVector * (numerator/denominator)
   # raise "Not yet implemented."
    # END_YOUR_CODE

############################################################
# Problem 3.3.2
    
def hingeLossGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of hinge loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of hinge loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    if(1-np.sum(theta*featureVector)*(2*y-1))>0:
        return -featureVector*(2*y-1)
    else:
        return featureVector*0
    #raise "Not yet implemented."
    # END_YOUR_CODE

