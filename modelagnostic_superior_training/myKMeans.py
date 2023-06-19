'''
This module implements k-means clustering via Lloyd's algorithm with some extensions.

E.g., it includes the distribution preserving version as proposed in "Distributional Clustering: A distribution-preserving clustering method" by (Krishna et. al, 2019)
https://arxiv.org/abs/1911.05940
'''

import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.utils import check_random_state
import numpy as np

# idea: write a kMeans procedure, when there are already some centers fixed (e.g. samples that were drawn according to an AL scheme in previous iterations)
# since I have already made sure to remove the fixed centers from the candidates X, we do not need indices of the fixed centers, and also do not need to remove the fixed centers from X 

def myKMeans(X, n_centers, densityX = None, targetDensityX = None, perms = None, n_init = 10, init_centers = 'k-means++', distributionalClustering = True, max_iter = 300, intrinsicDim = None, fixedCenters = None): # 'random', 'k-means++'
    x_squared_norms = row_norms(X, squared=True)
    
    if init_centers == 'k-means++':
        initCenters = _kmeans_plusplus
    if init_centers == 'random':
        initCenters = randomCenters
        
    if targetDensityX is None or densityX is None:
        importanceWeights = np.ones(len(X))
    else:
        importanceWeights = targetDensityX / densityX
        importanceWeights /= np.mean(importanceWeights)

    bestInertia = np.Inf
    for rep in range(n_init):
        # repeat initializations
        centers, inxes = initCenters(X, n_centers, densityX, targetDensityX, x_squared_norms=x_squared_norms, perms=perms, intrinsicDim = intrinsicDim, fixedCenters = fixedCenters)
        if max_iter == 0:
            # estimate one iteration, which gives lastInertia
            _,_,_,lastInertia = myLloydStep(X, centers, inxes, perms=perms, x_squared_norms = x_squared_norms, distributionalClustering = distributionalClustering, fixedCenters = fixedCenters, importanceWeights = importanceWeights)
        else:
            iteration = 0
            xClusterAssignmentOld = np.full(len(X), -2, dtype=np.int32)
            xClusterAssignment = np.full(len(X), -1, dtype=np.int32)
            while iteration < max_iter and not np.equal(xClusterAssignmentOld, xClusterAssignment).all():
                xClusterAssignmentOld = xClusterAssignment
                centers, inxes, xClusterAssignment, lastInertia = myLloydStep(X, centers, inxes, perms=perms, x_squared_norms = x_squared_norms, distributionalClustering = distributionalClustering, fixedCenters = fixedCenters, importanceWeights = importanceWeights)
                iteration += 1

        if lastInertia < bestInertia:
            bestInertia = lastInertia
            bestInxes = inxes
    return bestInxes, bestInertia
    
def myLloydStep(X, centers, cInxes, perms = None, x_squared_norms = None, distributionalClustering = True, fixedCenters = None, importanceWeights = None):

    # if we got fixed centers, we need to incorporate them when calculating the individual cluster supports, even though we do not update them!
    n_adaptiveCenters = len(centers)
    if not fixedCenters is None:
        centers = np.vstack((centers, fixedCenters))
    
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    newCenterInxes = np.full(n_adaptiveCenters, -1, dtype=np.int32)
    if perms is None:
        distance_centers_to_X = euclidean_distances(centers, X, Y_norm_squared=x_squared_norms,squared=True)
    else:
        permInxes = np.full([len(centers),len(X)], -1, dtype=np.int32)
        distance_centers_to_X = np.zeros([len(centers),len(X)]) + np.Inf
        for p, currentPerm in enumerate(perms):
            distOld = distance_centers_to_X
            distance_centers_to_X = np.minimum(distance_centers_to_X, euclidean_distances(centers, X[:,currentPerm], Y_norm_squared=x_squared_norms,squared=True))
            permInxes[distance_centers_to_X < distOld] = p
            
    clusterAssignment = np.argmin(distance_centers_to_X, axis=0)

    if not perms is None:
        permInxOfNearestCenter = permInxes[clusterAssignment, np.arange(len(X))]

    # for each cluster, average over the associated elements of X, in the representation that is closest to the center
    # also estimate the last center's inertia
    lastInertia = 0.
    for c in range(len(centers)):
        Xinxes = np.where(clusterAssignment == c)[0]
        if len(Xinxes) == 0:
            # Note: Can only happen to old fixedCenters
            continue
        clusterXes = X[Xinxes]
        clusterIWs = importanceWeights[Xinxes].ravel()
        if not perms is None:
            # permute Xes to representation closest to center
            clusterXesPermInxes = permInxOfNearestCenter[clusterAssignment == c]
            for p, currentPerm in enumerate(perms):
                if sum(clusterXesPermInxes == p) > 0:
                    clusterXes[clusterXesPermInxes == p] = clusterXes[clusterXesPermInxes == p][:, currentPerm]
                    
        if distributionalClustering:
            if c < n_adaptiveCenters:
                # virtually delete respective candidate from the cluster elements by setting its self-distance to 1
                distance_centers_to_X[c,cInxes[c]] = 1.
            if distance_centers_to_X[c,Xinxes].min() <= 0:
                # need numerically more stable, but more expensive distance estimate
                #distance_centers_to_X[c,Xinxes] = np.sqrt(np.square(centers[[c]][:, None, :] - X[Xinxes][None, :, :]).sum(-1))
                distance_centers_to_X[c,Xinxes] = np.square(centers[[c]][:, None, :] - X[Xinxes][None, :, :]).sum(-1)
                if c < n_adaptiveCenters:
                    distance_centers_to_X[c,cInxes[c]] = 1.
            lastInertia += np.sum(np.log(np.sqrt(distance_centers_to_X[c,Xinxes])) * clusterIWs)
                
            if c < n_adaptiveCenters:
                #TODO: more clever subset choice is possible: get inxes of 10% of samples nearest to cluster mean
                newCandidateCenterInxes = np.arange(len(Xinxes))
                candidateToClusterXesDists = euclidean_distances(clusterXes[newCandidateCenterInxes], clusterXes, Y_norm_squared=x_squared_norms[Xinxes],squared=False)
                # virtually delete respective candidate from the cluster elements by setting its self-distance to 1
                candidateToClusterXesDists[np.arange(len(newCandidateCenterInxes)),newCandidateCenterInxes] = 1.
                if candidateToClusterXesDists.min() <= 0:
                    # need numerically more stable, but more expensive distance estimate
                    candidateToClusterXesDists = np.sqrt((np.square(clusterXes[newCandidateCenterInxes][:, None, :] - clusterXes[None, :, :])).sum(-1))
                    candidateToClusterXesDists[np.arange(len(newCandidateCenterInxes)),newCandidateCenterInxes] = 1.

                candidateClusterInertias = (np.log(candidateToClusterXesDists) * clusterIWs).sum(1)
                # choose new center of the cluster
                newCenterInx = Xinxes[np.argmin(candidateClusterInertias)]
                # update center
                newCenterInxes[c] = newCenterInx
                centers[c] = X[newCenterInx]
        else:
            lastInertia += np.sum(distance_centers_to_X[c,Xinxes] * clusterIWs)
            if c < n_adaptiveCenters:
                clusterMean = (clusterXes * clusterIWs.reshape(-1,1)).mean(0).reshape(1,-1)
                # which element of clusterXes is closest to the new center?
                distsToNewCenter = euclidean_distances(clusterMean, clusterXes, Y_norm_squared=x_squared_norms[Xinxes],squared=True)
                newCenterInx = Xinxes[np.argmin(distsToNewCenter)]

                # update center
                newCenterInxes[c] = newCenterInx
                centers[c] = clusterMean
    return centers[:n_adaptiveCenters], newCenterInxes, clusterAssignment, lastInertia

def _kmeans_plusplus(X, n_centers, densityX = None, targetDensityX = None, x_squared_norms = None,
                     random_state=None, n_local_trials=None, perms = None, intrinsicDim = None, fixedCenters = None):

    random_state = check_random_state(random_state)
    
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    
    n_samples = X.shape[0]
    if intrinsicDim is None:
        intrinsicDim = X.shape[1]
        
    if targetDensityX is None:
        targetDensityX = densityX
        
    if targetDensityX is None:
        potentialDensityFactor = 1.
    else:
        #when halfing intersample distances:
        #- the density grows as 2**d
        #standard-algo: sample proportional to squared-distance-to-nearest-center (SD)
        #if X is uniformly distributed, the sample of centers will be uniformly distributed as well
        #if we want the centers to reflect the density, e.g. for p(x) = 2**d and p = 1 elsewhere, then we can modify the potential SD to (SD * P**(2/d))
        #reason: if we double the potential locally in x, we require 2**d chosen centers near x to obtain a comparable potential in x than elsewhere
        potentialDensityFactor = targetDensityX.reshape(1,n_samples)**(2/intrinsicDim)
        potentialDensityFactor /= np.mean(potentialDensityFactor)


    centers = np.empty((n_centers, X.shape[1]), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(len(centers)))

    indices = np.full(n_centers, -1, dtype=int)
    
    if fixedCenters is None:
        # Pick first center randomly and track index of point
        center_id = random_state.randint(n_samples)
        if sp.issparse(X):
            centers[0] = X[center_id].toarray()
        else:
            centers[0] = X[center_id]
        indices[0] = center_id

        # Initialize list of closest distances and calculate current potential
        if perms is None:
            closest_dist_sq = euclidean_distances(
                centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
                squared=True)
        else:
        ########## take nearest representer over perms as the closest distance
            closest_dist_sq = np.zeros([1,n_samples]) + np.Inf
            for currentPerm in perms:
                closest_dist_sq = np.minimum(closest_dist_sq, euclidean_distances(centers[0, np.newaxis], X[:,currentPerm], Y_norm_squared=x_squared_norms,squared=True))
                
    else:
        # begin by getting the closest dists with respect to all previously fixed centers
        closest_dist_sq = np.zeros([1,n_samples]) + np.Inf
        for fc in fixedCenters:
            if perms is None:
                closest_dist_sq = np.minimum(closest_dist_sq, euclidean_distances(fc.reshape(-1,1), X, Y_norm_squared=x_squared_norms,squared=True))
            else:
            ########## take nearest representer over perms as the closest distance
                for currentPerm in perms:
                    closest_dist_sq = np.minimum(closest_dist_sq, euclidean_distances(fc.reshape(-1,1), X[:,currentPerm], Y_norm_squared=x_squared_norms,squared=True))
    
    individualPots = closest_dist_sq * potentialDensityFactor
    current_pot = individualPots.sum()
    
    # Pick the remaining n_centers-1 points ( or n_centers, when already fixed centers exist)
    for c in range((1 if fixedCenters is None else 0), n_centers):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(individualPots),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, individualPots.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        if perms is None:
            distance_to_candidates = euclidean_distances(
                X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)
        else:
            distance_to_candidates = np.zeros([len(candidate_ids),n_samples]) + np.Inf
            for currentPerm in perms:
                distance_to_candidates = np.minimum(distance_to_candidates, euclidean_distances(X[candidate_ids], X[:,currentPerm], Y_norm_squared=x_squared_norms,squared=True))

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        
        individualCandidatePots = distance_to_candidates * potentialDensityFactor
        
        candidates_pot = individualCandidatePots.sum(axis=1)
        
        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate].reshape(1,n_samples)
        individualPots = closest_dist_sq * potentialDensityFactor
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices

def randomCenters(X, n_centers, densityX = None, targetDensityX = None, x_squared_norms = None, random_state=None, n_local_trials=None, perms = None, intrinsicDim = None, fixedCenters = None):

    if densityX is None or targetDensityX is None:
        resamplingWeights = np.ones(len(X))
    else:
        resamplingWeights = targetDensityX / densityX
    resamplingWeights /= np.sum(resamplingWeights)
    
    indices = np.random.choice(len(X), n_centers, replace=False, p = resamplingWeights)
    return X[indices], indices
