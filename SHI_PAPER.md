Early burst detection for memory-efficient image retrieval
Miaojing Shi, Yannis Avrithis, Hervé Jégou
To cite this version:
Miaojing Shi, Yannis Avrithis, Hervé Jégou. Early burst detection for memory-efficient image retrieval. Com-
puter Vision and Pattern Recognition, Jun 2015, Boston, United States. ⟨hal-01146533⟩
HAL Id: hal-01146533
https://inria.hal.science/hal-01146533v1
Submitted on 28 Apr 2015
HAL is a multi-disciplinary open access archive
for the deposit and dissemination of scientific re-
search documents, whether they are published or not.
The documents may come from teaching and research
institutions in France or abroad, or from public or pri-
vate research centers.
L’archive ouverte pluridisciplinaire HAL, est des-
tinée au dépôt et à la diffusion de documents scien-
tifiques de niveau recherche, publiés ou non, émanant
des établissements d’enseignement et de recherche
français ou étrangers, des laboratoires publics ou
privés.
HAL Authorization
Early burst detection for memory-efficient image retrieval
Miaojing Shi∗
Peking University
Yannis Avrithis
University of Athens, NTUA
Herv´ e J´
egou
Inria
Abstract
Recent works show that image comparison based on lo-
cal descriptors is corrupted by visual bursts, which tend to
dominate the image similarity. The existing strategies, like
power-law normalization, improve the results by discount-
ing the contribution of visual bursts to the image similarity.
In this paper, we propose to explicitly detect the visual
bursts in an image at an early stage. We compare several
detection strategies jointly taking into account feature sim-
ilarity and geometrical quantities. The bursty groups are
merged into meta-features, which are used as input to state-
of-the-art image search systems such as VLAD or the selec-
tive match kernel. Then, we show the interest of using this
strategy in an asymmetrical manner, with only the database
features being aggregated but not those of the query.
Extensive experiments performed on public benchmarks
for visual retrieval show the benefits of our method, which
achieves performance on par with the state of the art but
with a significantly reduced complexity, thanks to the lower
number of features fed to the indexing system.
1. Introduction
SELF-SIMILARITY in images is a recurring concept in
image analysis and computer vision. It is related to
the statistics of natural images, which are highly redundant.
This concept has been tackled in many applications. For
instance, it is exploited in the non-local means de-noising
algorithm [6], where each patch is updated by a weighted
sum of similar patches. It is also useful for spatial verifi-
cation and image rectification by exploiting the geometri-
cal assumption of repeated patterns [28]. In the context of
image matching, early works [10] like the one of Shecht-
man and Irani [33] compute the similarity between two im-
ages with the “self-similarity descriptor”, a compact local
descriptor that is densely computed on the image.
In recent works based on SIFT local features [19], the
underlying statistical phenomenon is often referred to as vi-
sual burstiness [14], by analogy to the terminology used in
∗Miaojing Shi has worked on this paper while he was a visiting student
at INRIA Rennes.
a context of textual information retrieval. As in [18, 20]
where models like the Dirichlet distributions [20] are pro-
posed to better reflect the bursty nature of documents.
Computer vision researchers question the issue of fea-
ture independence implicitly assumed in the bag-of-words
model [14, 17, 8, 31, 41, 9], they show the importance of
taking care of visual burstiness in image retrieval and im-
age classification: the bursty features tend to dominate the
similarity measure, which degrades the quality of the com-
parison, as other non-bursty yet possibly distinctive features
have a comparatively lower contribution.
Various strategies have been proposed to discount the
contribution of bursts on the similarity measure. Among
the proposed approaches, some are inspired by text like
the so-called power-law normalization [14, 17] for bag-of-
words or the Polya or Dirichlet models [8]. These strategies
have been pragmatically extended to and improved for more
complex image vector representations such as VLAD [2, 9]
or the Fisher vector [25, 8]. They are also standardly used
in matching approaches like Hamming Embedding [15] or
selective match kernels (SMK/ASMK) [37]. Recently, We
stress a key difference between textual and visual bursts. In
images, the space of features is continuous and not discrete
as in text, which makes it difficult to determine which fea-
tures are bursty or not. This may explain why bursts are han-
dled by very simple techniques such as power-law, residual
normalization [2, 37] or weighting votes [14].
This paper revisits the concept of visual bursts by con-
sidering several detection strategies aiming at identifying
the bursts directly from the SIFT descriptors. Our approach
is inspired by the work of Turcot and Lowe [42], who re-
move the features that are unlikely to match a query in an
image collection. In contrast to their approach, which re-
quires to off-line cross-match the whole database, we focus
on detecting bursts within an image. Another difference is
that we merge similar descriptors into a single representa-
tive vector instead of selecting features. Therefore we com-
press the representation without discarding any feature.
The effectiveness of our approach depends on two im-
portant design choices. First, we construct the affinity ma-
trix between patches. We combine (i) a probabilistic mea-
sure learned on a patch database [44] with (ii) a kernel de-
1
fined on geometrical quantities such as scale and orienta-
tion, which were shown useful for burst detection by Torii
et al. [41]. Second, we evaluate several kernelized cluster-
ing algorithms to produce groups from the affinity matrix.
The clear winner amongst all clustering methods is a sim-
ple strategy performing the connected component analysis
of the thresholded matrix.
Finally, we propose an asymmetric aggregation method:
we apply our burst detection method on database side but do
not aggregate the query descriptors. This further improves
the performance while offering a memory footprint per im-
age identical to the one of our symmetric burst aggregation.
The paper is organized as follows. Section 2 introduces
our representation and retrieval model. Section 3 discusses
the concept of visual burst and proposes our method to de-
tect bursts. Section 4 evaluates our methods on public re-
trieval benchmarks. Our results demonstrate the benefits of
our approach, which improves the state-of-the-art in mem-
ory/performance trade-off.
2. Representation and matching
We assume an image is represented by a set Fof lo-
cal features, where each feature f ∈Fis represented by
a d-dimensional local descriptor uf, local scale sf and ori-
entation θf. We ignore position because we do not use ge-
ometrical information to match two images, but rather to
analyze similarities within images. We further assume that
each descriptor x = uf is quantized on a codebook Cof k
visual words, or cells. We adopt the matching model [37],
whereby the similarity of images F,Gis measured by
S(X,Y) = ν(X)ν(Y)
wcM(Xc,Yc), (1)
c∈C
where X,Yare the descriptors of F,Grespectively, Xc are
the descriptors of X assigned to cell c, M is a cell sim-
ilarity function, wc is a weighting factor for c and ν(X)
is a normalization factor such that self-similarity of X is
S(X,X) = 1. Although (1) is a general model that in-
cludes special cases of several popular methods like bag of
words (BoW) [35], VLAD [17] and Hamming Embedding
(HE) [13], it is clearly motivated by discarding geometric
information for efficiency reasons.
We would rather like to detect burstiness at an early stage
to fuse them, and to use any standard search infrastructure
off-the-shelf. Therefore, given an image F, we detect fea-
ture bursts in Fand represent it by a set of aggregated de-
scriptors. This is an off-line operation, i.e., the descriptors
can be quantized, encoded and searched as usual.
For the cell similarity function M, we use VLAD [17, 17]
and SMK/ASMK [37]. These are two methods targeted for
different scenarios with different memory/speed/accuracy
compromises. Our descriptor aggregation can be combined
Figure 1. An image along with the features of the six most popu-
lated bursts detected. A dot is shown at the position of each fea-
ture, colored according to the burst it belongs to; the remaining
features are not shown.
0.2
principal component 12
0.1
0
−0.1
−0.2
−0.4−0.2 0 0.2
principal component 6
Figure 2. The 1610 features of the image of Fig. 1 in descriptor
space. After PCA analysis, we plot two of the principal compo-
nents found (component 6 vs. 12). Colored groups of points corre-
spond to the six most populated bursts, exactly as in Fig. 1. Gray
points correspond to smaller bursts. Smaller light gray dots are
isolated descriptors that do not belong to any burst.
and provide benefits in both frameworks. Interestingly, the
state-of-the-art ASMK, which is also an aggregated rep-
resentation intended to address the burstiness problem, is
complementary to our strategy.
3. Early detection of visual bursts
What are bursts? As a representative example, Fig. 1 de-
picts a clean view of a building exhibiting a lattice structure
of almost identical tiles formed by its windows. To visualize
the resulting groups of similar patches, we extract normal-
ized RootSIFT [1] descriptors on Hessian-affine [21] fea-
tures, compute pairwise distances, and find the connected
components formed by joining pairs whose distance is be-
low a certain threshold. This is a simple but very effective
approach. By coloring features according to their group, it
is clearly seen that the same pattern of 5-6 groups appears
around every window on the building surface.
Since patches are very similar in appearance, one might
expect a high density of points around each burst in the de-
scriptor space, so that bursts can be easily found by clus-
tering or mode seeking. However, as illustrated in Fig. 2,
this is far from being true. It is not easy to visualize what
happens in a 128-dimensional space by a mere 2D projec-
tion. Still, whatever the projection, isolated descriptors (not
belonging to any group) appear to have the same density as
bursty ones (belonging to some group), while bursts have ar-
bitrary shape and large extent. One cannot hope that bursts
will fit within the cells of a codebook, especially if the the
latter is trained on random samples without taking bursti-
ness into account. Our approach to detect bursts before
quantizing descriptors then makes sense.
Besides structure in man-made scenes, other sources of
burstiness may be texture, e.g. in natural environments, or
the fact that feature detectors may give multiple responses,
e.g. along edges or around corners at different scales. De-
pending on the case, geometry also plays a role. In contrast
to methods that focus on repeating structures and symme-
tries [41, 40], we consider individual features disregarding
position and neighborhood: the search for repeating groups
of features is too constrained and may fail to identify bursts,
especially in natural scenes. However, we do investigate
scale and orientation. For instance, similar patches in Fig. 1
have the same orientation, but this is not always the case on
a textured surface. It is not known whether bursty features
share the same appearance only or scale and orientation as
well; this is something we determine experimentally.
Section 3.1 defines a feature kernel measuring the sim-
ilarity of two individual features. Then in section 3.2 we
consider a number of methods to detect bursts using the
given kernel function and conduct a preliminary evaluation.
3.1. Feature kernel
Given two local features f,g in an image, we define a
feature kernel function
k(f,g) = ku(uf,ug)ks(sf,sg)kθ(θf,θg), (2)
consisting of three factors, namely the descriptor kernel ku,
the scale kernel ksand the orientation kernel kθ. Intuitively,
this function measures if the two corresponding patches in
the image are similar in appearance and have similar scales
and orientations. Factors ks,kθ can be optionally omitted.
Descriptor kernel. Given a pair of descriptors x,y ∈Rd
,
descriptor kernel ku measures their similarity and is a func-
tion of the inner product z = ⟨x,y⟩, which is equivalent
to their distance if x,y are ℓ2-normalized. Seeing z as a
random variable, we define this function as the probability,
given z, that x,y belong to the same burst, i.e. they cor-
respond to two matching image patches. In particular, we
adopt a generative model for a binary classifier: if Bis the
class of descriptor pairs that belong to the same burst and B
is its complement, we define
ku(x,y) = p(B|⟨x,y⟩), (3)
where
p(B|z) = p(z|B)p(B)
p(z|B)p(B) + p(z|B)p(B) (4)
is the posterior probability of Bgiven z; p(z|B),p(z|B)
are the class-conditional densities of B,Brespectively and
p(B),p(B) are their prior probabilities.
We train such a classifier from a dataset of
matching/non-matching patch pairs [45]. This dataset
consists of patches sampled from 3D reconstructions of
the Statue of Liberty (New York), Notre Dame (Paris) and
Half Dome (Yosemite). Two patches in two different views
of the same 3D scene are matched if they are projections
of the same 3D point. Such patches are very similar in
appearance, so they provide a good model for bursts.
We extract a SIFT descriptor from each patch in the
dataset and compute the inner product z = ⟨x,y⟩for each
pair (x,y) of descriptors. If B,B are the sets of all obser-
vations of z for matching and non-matching pairs respec-
tively according to the ground truth, we model the class-
conditional densities p(z|B),p(z|B) by fitting normal den-
sities to the samples of B,B respectively, according to
maximum likelihood. The prior probabilities are p(B) =
|B|/N,p(B) = |B|/N, where N= |B|+ |B|is the total
number of samples. Fig. 3 shows the posterior probabil-
ity p(B|z) computed from (4). It appears that ku(x,y) is a
sigmoid function of the inner product z = ⟨x,y⟩that can
discriminate well enough matching from non-matching de-
scriptor pairs.
Scale and orientation kernels. Given two patch scales s,t,
scale kernel ks(s,t) is given by a Gaussian kernel of their
logarithms,
ks(s,t) = exp−λlog2 s
t
. (5)
This choice makes ks invariant to absolute scale, as long as
ratio s/tremains constant.
The situation is similar for orientation, but following the
work [38], we use the equivalent of Gaussian for periodic
distributions, which is the von Mises distribution. In par-
ticular, given two patch orientations θ,φ, orientation kernel
kθ(θ,φ) is given by the von Mises kernel
kθ(θ,φ) = exp(κcos(θ−φ))−exp(−κ)
2 sinh(κ). (6)
Parameters λ,κare tuned experimentally according to [38].
8
p(z|B)
6
p(z|B), p(z|B)
p(z|B)
p(B|z)
4
2
0
1
0.8
0.6
0.4
0.2
0
p(B|z)
0.2 0.4 0.6 0.8 1
z
Figure 3. Distributions of z = ⟨x,y⟩for matching and non-
matching SIFT descriptor pairs (x,y) from the dataset of [45],
where we fit class-conditional densities p(z|B),p(z|B). Posterior
probability p(B|z) is used to compute kernel ku (3).
3.2. Burst detection and aggregation
Given an image F= {f1,...,fn}with nfeatures, de-
fine n×naffinity matrix Kwith elements
Kij= k(fi,fj) (7)
for i,j = 1,...,n, where kernel k is given by (2). The
affinity matrix includes all pairwise feature similarities in
Fand is the only input for burst detection. Now, given the
discussion so far, a candidate algorithm should
•be based on a kernel method, or more generally, op-
erate on metric spaces. One reason is that bursts are
of arbitrary shape in the feature space, as shown in
Fig. 2. Another is that we have formulated the input as
an affinity matrix, in order to combine with scale and
orientation proximity. This excludes algorithms that
represent points in a Euclidean space, like k-means.
•be able to automatically determine the number of
groups such that non-matching features are never
grouped, or at least have a parameter to control it so
that the resulting number varies smoothly and can be
close to the original size n. This is because we would
like the number of aggregated descriptors to vary from
ndown to a certain percentage of n.
Since we look for groups of similar features as measured
by kernel k, any clustering or mode seeking algorithm that
respects the above constraints would do in theory. Thus we
examine a number of existing methods.
Connected components. Features are treated as nodes of
an undirected graph with an edge for each pair (f,g) of fea-
tures with k(f,g) above a threshold τ. Then we compute
the connected components of the graph and consider each
component with more than one feature to be a bursty group.
The remaining components each contain one isolated fea-
ture. This is the fastest and most effective method.
Quick shift [43] is a very simple, fast mode seeking method
that can operate in non-Euclidean spaces. Eventually, all
points are connected into a single tree; after that, edges are
disconnected according to a threshold τ. This aspect is sim-
ilar to connected components: τ needs to be tuned to obtain
the desired number of groups, but we evaluate a range of
values in our experiments anyway.
Kernel k-means [32] is a simple kernel method,
parametrized by the desired number kof groups. However,
it has high complexity and is not designed for kbeing large.
Spectral methods. Representative spectral methods in-
clude spectral clustering [23] and normalized cuts [34, 11].
Certain group of bursty features is expected to contribute
most of its energy to one of the leading eigenvectors of ma-
trix Kand give the largest projection on this eigenvector. It
is possible to specify the number of groups — e.g., apply
k-means on those leading eigenvectors [23]. But the cost is
prohibitive in general. We design a method which hierar-
chically applies spectral clustering to subdivide each group,
while automatically determining the number of groups. We
call this variant hierarchical spectral clustering.
Evaluation. We conduct a preliminary qualitative evalua-
tion of the above methods on a limited sample of images to
determine whether they indeed satisfy our constraints and
make an initial selection before moving on to larger scale
quantitative experiments. We present here an example on
one representative image as shown in Fig. 4.
We have tuned each method such that the number of de-
tected groups is 80-85% of the initial features; since all
descriptors in a group are aggregated, this measure is re-
ferred as aggregation% in the experiments and is crucial in
the memory-performance trade-off. It can be seen that most
methods yield consistent groups; however, normalized cuts
gives many inconsistent groups while the groups found by
spectral clustering are too small. The result of connected
components is the cleanest, and Fig. 6 verifies this fact: each
group contains almost identical patches, precisely centered
at the position of each feature.
Fig. 5 further shows the group size distributions. It ap-
pears that kernel k-means is problematic in the sense that
the groups found are far less than the target, resulting in an
aggregation% of 39%, rather than 80%. Although the re-
quired number of groups is given as a parameter, too many
clusters become empty and the few remaining are most de-
tected as bursts. Spectral clustering finds too many small
groups. In general, we would like most features to remain
isolated and the remaining ones form few bursts. In this
sense, connected components, hierarchical spectral cluster-
ing and normalized cuts appear reasonable.
feature set #features threshold patch size
Holidays-S 4.4M 500 21
Holidays-M 4.3M 500 41
Holidays-L 6.6M 300 41
Table 1. Three feature sets obtained from Holidays with different
detector parameters. Holidays-S is the default.
Figure 6. The six most populated bursts found by connected com-
ponents on the example of Fig. 4. An image patche of size 30 ×30
pixels is shown for each feature; a dot is shown at each feature
position, colored according to burst exactly as in Fig. 4.
Taking into account the preliminary evaluation and the
fact that the spectral methods are quite slow to apply at large
scale, we choose the first three methods of Fig. 4 for further
quantitative evaluation in the context of retrieval.
Burst aggregation. Given an image, the result of burst de-
tection is a partition of its features into groups. We simply
take the average of the descriptors in each group and ℓ2-
normalize them. Discarding geometry, this yields a set of
(aggregated) descriptors to represent the image, so any en-
coding or search model applies.
4. Experiments
We evaluate and compare the proposed burst detection
and aggregation in the context of two different image re-
trieval models, namely VLAD and SMK/ASMK. We first
discuss the evaluation protocol and give some implemen-
tation details; then we discuss the impact of parameters
for each method and analyze the benefits we obtain in the
memory-performance trade-off.
4.1. Experimental setup
Datasets. We conduct experiments on three retrieval bench-
marks, namely Holidays [13], Oxford [26], and Paris [27].
We study the impact of parameters mainly on Holidays. To
evaluate performance on larger scale, we add distractors
from the Flickr 100k set [26] to Holidays and Oxford.
Descriptors. Local features are extracted with the Hessian-
affine detector [21] on Holidays and its improved ver-
sion [24] on Oxford and Paris. We adopt the default pa-
rameters of the detector, but we also use a lower threshold
and larger patch size to yield different feature sets for Hol-
idays. Since we are generating a reduced feature set by ag-
gregating, this helps evaluate the performance at the same
memory depending on the initial feature set. Table 1 shows
the three different feature sets used. We use RootSIFT [1]
descriptors in all our experiments.
Evaluation. Retrieval performance is measured in terms
of mean average precision (mAP). As we vary the number
of detected bursts, we generate aggregated feature sets of
varying size; the ratio to the original size, averaged over
a dataset, is called aggregation%. We thus measure mAP
as a function of aggregation% to evaluate the memory-
performance trade-off. When an inverted file is used, we
also measure the imbalance factor [36], which is directly
related to the search cost and should be as close as possible
to the optimal value of 1.
Burst detection and aggregation. On Holidays, we em-
ploy the proposed descriptor kernel and scale kernel only;
referring to (2), we use kernel kuks. On Oxford and Paris
on the other hand, we use all three kernels, i.e. descriptor,
scale and orientation; that is, kukskθ. This setting always
gives the best performance. We initially evaluate three dif-
ferent burst detection methods, and then focus on connected
components. We apply different thresholds to the affinity
matrix to vary the number of bursts such that aggregation%
varies in the range of 10-100%. In all graphs, the baseline
is always the rightmost point. We follow two strategies:
database descriptors are always aggregated, while query de-
scriptors may be aggregated or not; these are called sym-
metric and asymmetric aggregation, respectively.
Retrieval models. We conduct experiments on two
representative retrieval models, VLAD [17] and
SMK/ASMK [37]. In particular, we use the efficient
versions SMK*/ASMK* where descriptors are binarized.
Both models target at reducing the effect of burstiness:
VLAD by power-law normalization, and ASMK by its own
aggregation after quantization. We present the benefit from
our early burst detection but interestingly we also show it
can be complementary to such methods.
Cost. The query cost is quadratic (linear) in aggregation%
for symmetric (asymmetric) aggregation, i.e. always less
than baseline. The off-line additional cost of burst detection
and aggregation is O(n2) where nis the number of features
per image. As a comparison, quantization with a flat vocab-
ulary is O(nk), where k is the vocabulary size. Therefore
the fixed cost of early detection is negligible in the AMSK
pipeline (k ≫n) and more expensive for VLAD (k < n),
yet reasonable.
connected components quick shift hierarchical spectral clustering
kernel k-means normalized cuts spectral clustering
Figure 4. Feature grouping and burst detection with six methods. In each case, the six most populated bursts are shown with a dot at the
position of each feature, colored according to the burst it belongs to.
connected components
quick shift
hierarchical spectral clustering
group size
isolated
bursts
group size
6
4
2
0
isolated
bursts
group size
isolated
bursts
100 101 102 103
group #
kernel k-means
100 101 102 103
group #
normalized cuts
100 101 102 103
group #
spectral clustering
isolated
bursts
isolated
bursts
group size
isolated
bursts
15
15
10
10
5
5
0
0
3
15
group size
10
5
group size
0
10
8
6
4
2
0
2
1
0
100 101 102 103
100 101 102 103
100 101 102 103
group #
group #
group #
Figure 5. Size distribution of groups found by each method for the example of Fig. 4. Groups of two features or more are considered bursts
and the remaining groups each contain one isolated feature. Observe the logarithmic axis: bursts are only a small fraction of groups.
4.2. Results on VLAD
Burst detection and aggregation. Fig. 7 illustrates the per-
formance of VLAD on Holidays-L under varying aggrega-
tion%. As discussed in Section 3.2, we compare the three
most promising methods of the initial qualitative evalua-
tion, i.e. connected components, quick shift and hierarchi-
cal spectral clustering. Connected components choice is al-
ways superior, while the other two methods do not achieve
any benefit. Therefore, we focus all remaining experiments
on connected components.
Large-scale. Table 2 shows large scale results on Holidays-
L plus distractors. Recall that aggregation% of 1 refers to
the baseline. It is impressive that we get absolute perfor-
mance gain at reduced memory and query time.
aggregation% 1.000 0.764 0.638 0.556
k = 16 41.3 42.7 44.1 45.0
k = 64 46.3 47.5 48.3 48.8
Table 2. VLAD mAP performance vs. aggregation% on Holidays-
L +Flickr 100k distractors for two vocabulary sizes, 16 and 64.
4.3. Results on SMK*/ASMK*
Symmetric vs. asymmetric. Fig. 8 compares symmetric
and asymmetric aggregation on Holidays-L; recall that in
the latter case the query is not aggregated. It turns out that
for low aggregation% asymmetric is superior, and this ob-
servation holds for all our experiments, so we limit to this
55
mAP
50
45
connected components
quick shift
hierarchical spectral clustering
0 0.2 0.4 0.6 0.8 1
aggregation%
Figure 7. VLAD performance vs. aggregation% on Holidays-L for
three burst detection methods. Vocabulary size k = 16; baseline
power law parameter α= 1.
strategy for the remaining results. This behavior can be ex-
plained by the fact that under severe aggregation on both
images, most matches are lost. We also observe an im-
pressive improvement on the memory-performance trade-
off: we can keep only 30% of the original descriptors for a
performance drop of merely 1%.
Initial features. Fig. 9 compares three different initial fea-
ture sets on Holidays and measures mAP vs. absolute num-
ber of descriptors/image, which directly reflects memory.
Now comparing the three sets for any number of descrip-
tors, the largest set maintains a gain of over 10% over the
smallest one. This is another aspect of the trade-off and
suggests a way to improve performance: augment the initial
features, aggregate, and gain in mAP at the same memory.
Imbalance factor. Fig. 10 and11 investigate the imbalance
factor [36] on Holidays-L, Oxford and Paris. By aggre-
gating bursty features at an early stage, we make the in-
verted file more balanced. Interestingly, the imbalance fac-
tor exhibits a minimum at an aggregation% which gives at
the same time a good memory-performance trade-off, e.g.
60%, 30% respectively for SMK*/ASMK* on Holidays-L.
In terms of query cost, the benefit of improved imbalance
factor should be multiplied by the benefit due to decreased
memory: indeed, query time is linear in aggregation%.
Large scale. Table 3 shows large scale results on Holidays-
L and Oxford plus distractors. It is interesting that e.g. in
Oxford, the result is more promising than at small scale:
we can save 15% of memory at no performance cost and
increase efficiency at the same time.
Comparison to the state of the art. Table 4 shows state-
of-the-art results compared to our best results on ASMK*.
We only compare to methods relating to vocabularies and
descriptor representation and not e.g. spatial matching [24,
4], query expansion [7, 39], feature augmentation [42, 1] or
nearest neighbor re-ranking [29].
85
mAP
80
75
SMK*, symmetric
ASMK*, symmetric
SMK*, asymmetric
ASMK*, asymmetric
0.2 0.4 0.6 0.8 1
aggregation%
Figure 8. SMK*/ASMK* performance vs. aggregation% on
Holidays-L for symmetric and asymmetric aggregation. Vocab-
ulary size k= 65k; selectivity exponent α= 3.
dataset Holidays-L 101k Oxford 105k
aggregation% 0.65 0.52 0.28 0.90 0.76 0.55
mAP 85.1 84.5 77.6 68.9 68.9 63.6
Table 3. ASMK* mAP performance on Holidays-L and Oxford
plus Flickr 100k distractors. Vocabulary size: 65k. The first col-
umn of each dataset is the baseline.
The first group of methods relies on a large vocabulary
(1M or more) and does not include a descriptor signature.
Performance may be improved by learning a finer vocabu-
lary on a larger training set [22], which is a costly off-line
process, or using the extremely fine partition of a multi-
index [3, 5], which cannot be fully inverted. The second
group relies on a smaller vocabulary (100k or less) and em-
beds a descriptor signature, e.g. a Hamming code [12, 37]
as in this work, or product quantization code [30, 16]. This
approach is superior, but requires additional space.
The third group includes ASMK* [37] and this work.
There is still a descriptor signature, but the number of de-
scriptors is reduced, as indicated by aggregation%, which
is different for each dataset. Despite the lower memory
and faster query, these methods are superior to previous
ones. Additionally, we get a performance gain over [37]
using multiple assignment; in particular, the five nearest vi-
sual words as used in [37]. In Holidays, we start from the
larger feature set Holidays-L and aggregate such that the
total number of features is not higher than in [37], as in
Fig. 9. In the remaining datasets, the gain is due to absolute
improvement in the default feature set.
5. Conclusion
Handling burstiness has a significant and positive impact
on the accuracy of image search comparison. This paper
has shown that our early detection of visual bursts as an
effective strategy to produce image match kernel based on
local descriptors. It is applied prior to the indexing stage,
which might separate bursty features if quantization is em-
85
80
mAP
75
70
Holidays-S, α= 3
Holidays-S, α= 4
Holidays-S, α= 5
Holidays-M, α= 3
Holidays-M, α= 4
Holidays-M, α= 5
Holidays-L, α= 3
Holidays-L, α= 4
Holidays-L, α= 5
1,000 2,000 3,000 4,000
#descriptors/image
Figure 9. ASMK* performance vs. average number of aggregated
descriptors per image on Holidays for three different initial feature
sets and different values of selectivity exponent α. Vocabulary
size k = 65k; asymmetric aggregation. Note that the rightmost
measurement corresponds to aggregation% less than 1.
imbalance factor
3
2.5
2
1.5
Holidays-L, SMK*
Holidays-L, ASMK*
0.2 0.4 0.6 0.8 1
aggregation%
Figure 10. SMK*/ASMK* imbalance factor vs. aggregation% on
Holidays-L. Vocabulary size k= 65k; selectivity exponent α= 3.
1.25
Oxford, SMK*
Oxford, ASMK*
Paris, SMK*
Paris, ASMK*
imbalance factor
1.2
1.15
0.2 0.4 0.6 0.8 1
aggregation%
Figure 11. SMK*/ASMK* imbalance factor vs. aggregation% on
Oxford and Paris. Vocabulary size k = 65k; selectivity exponent
α= 3.
Dataset MA Hol. Paris Oxf.
BoW [27] - - 40.3
BoW [27] - - 49.3
BoW [24] - - 55.8
Fine vocab. [22] 74.9 74.9 74.2
Multi-index [3] - 69.6 70.3
HE [15] 74.5 - 51.7
HE [15] 77.5 - 56.1
AHE+burst [12] 79.4 - 66.0
AHE+burst [12] 81.9 - 69.8
Query ad. [30] 81.4 70.3 73.9
Query ad. [30] 82.1 73.6 78.0
aggregation% 78% 86% 89%
ASMK* [37] 80.0 74.4 76.4
ASMK* [37] 81.0 77.0 80.4
This work 88.1 77.5 81.3
Table 4. Comparison of our best mAP result to state-of-the-art us-
ing inverted files as in BoW or also local descriptors as in HE. We
only report results without spatial re-ranking.
ployed, and exploits geometrical quantities jointly with fea-
ture similarity. Not only our strategy is as or more effective
than other methods for handling visual bursts, but it is also
complementary to concurrent like the ASMK search engine,
leading to state-of-the-art results in a comparable setup.
Another key advantage is that, by fusing the descriptors
before feeding them to the indexing or search system, we
reduce the computational cost in both the quantization and
retrieval steps, typically by a factor of two. Our method
also reduces the memory footprint in the same proportion
for image search engines employing an inverted file.
Acknowledgements. This work was supported by ERC grant
VIAMASS no. 336054. Miaojing Shi was partially supported by
NBRPC 2011CB302400, NSFC 61121002, 61375026 and JCYJ
20120614152136201, and Yannis Avrithis was supported by the
EU (European Social Fund) and Greek National Fund through the
operational program “Education and Lifelong Learning” of the
National Strategic Reference Framework, research funding pro-
gram “ARISTEIA”, project “ESPRESSO”.
References
[1] R. Arandjelovic and A. Zisserman. Three things everyone
should know to improve object retrieval. In CVPR, 2012.
[2] R. Arandjelovic and A. Zisserman. All about VLAD. In
CVPR, 2013.
[3] Y. Avrithis. Quantize and conquer: A dimensionality-
recursive solution to clustering, vector quantization, and im-
age retrieval. In ICCV. 2013.
[4] Y. Avrithis and G. Tolias. Hough pyramid matching:
Speeded-up geometry re-ranking for large scale image re-
trieval. IJCV, 107(1):1–19, 2014.
[5] A. Babenko and V. Lempitsky. The inverted multi-index. In
CVPR, 2012.
[6] A. Buades, B. Coll, and J.-M. Morel. A non-local algorithm
for image denoising. In CVPR, 2005.
[7] O. Chum, A. Mikulik, M. Perdoch, and J. Matas. Total recall
II: Query expansion revisited. In CVPR, 2011.
[8] R. Cinbis, J. Verbeek, and C. Schmid. Image categoriza-
tion using Fisher kernels of non-iid image models. In CVPR,
2012.
[9] J. Delhumeau, P. Gosselin, H. J´ egou, and P. Perez. Revisiting
the VLAD image representation. In ACM Multimedia, 2013.
[10] T. Deselaers and V. Ferrari. Global and efficient self-
similarity for object classification and detection. In CVPR,
2010.
[11] C. Fowlkes, S. Belongie, F. Chung, and J. Malik. Spectral
grouping using the Nystr¨ om method. PAMI, 26(2):214–225,
2004.
[12] M. Jain, H. J´ egou, and P. Gros. Asymmetric hamming em-
bedding. In ACM Multimedia, 2011.
[13] H. J´ egou, M. Douze, and C. Schmid. Hamming embedding
and weak geometric consistency for large scale image search.
In ECCV, 2008.
[14] H. J´ egou, M. Douze, and C. Schmid. On the burstiness of
visual elements. In CVPR, 2009.
[15] H. J´ egou, M. Douze, and C. Schmid. Improving bag-of-
features for large scale image search. IJCV, 87(3):316–336,
2010.
[16] H. J´ egou, M. Douze, and C. Schmid. Product quantization
for nearest neighbor search. PAMI, 33(1):117–128, 2011.
[17] H. J´ egou, F. Perronnin, M. Douze, J. Sanchez, P. Perez, and
C. Schmid. Aggregating local image descriptors into com-
pact codes. PAMI, 34(9):1704–1716, 2012.
[18] D. Lewis. Naive (Bayes) at forty: The independence assump-
tion in information retrieval. In ECML, 1998.
[19] D. Lowe. Distinctive image features from scale-invariant
keypoints. IJCV, 60(2):91–110, 2004.
[20] R. E. Madsen, D. Kauchak, and C. Elkan. Modeling word
burstiness using the dirichlet distribution. In ICML, 2005.
[21] K. Mikolajczyk and C. Schmid. Scale & affine invariant in-
terest point detectors. IJCV, 60(1):63–86, 2004.
[22] A. Mikulik, M. Perdoch, O. Chum, and J. Matas. Learning a
fine vocabulary. In ECCV, 2010.
[23] A. Ng, M. Jordan, and Y. Weiss. On spectral clustering:
Analysis and an algorithm. In NIPS, 2002.
[24] M. Perdoch, O. Chum, and J. Matas. Efficient representation
of local geometry for large scale object retrieval. In CVPR,
2009.
[25] F. Perronnin, J. S´ anchez, and T. Mensink. Improving the
fisher kernel for large-scale image classification. In ECCV,
2010.
[26] J. Philbin, O. Chum, M. Isard, J. Sivic, and A. Zisser-
man. Object retrieval with large vocabularies and fast spatial
matching. In CVPR, 2007.
[27] J. Philbin, O. Chum, J. Sivic, M. Isard, and A. Zisserman.
Lost in quantization: Improving particular object retrieval in
large scale image databases. In CVPR, 2008.
[28] J. Pritts, O. Chum, and J. Matas. Detection, rectification and
segmentation of coplanar repeated patterns. In CVPR, 2014.
[29] D. Qin, S. Gammeter, L. Bossard, T. Quack, and
L. Van Gool. Hello neighbor: Accurate object retrieval with
k-reciprocal nearest neighbors. In CVPR, 2011.
[30] D. Qin, C. Wengert, and L. Van Gool. Query adaptive simi-
larity for large scale object retrieval. In CVPR, 2013.
[31] J. Revaud, M. Douze, and C. Schmid. Correlation-based
burstiness for logo retrieval. In ACM Multimedia, 2012.
[32] B. Sch¨ olkopf, A. Smola, and K. Muller. Nonlinear compo-
nent analysis as a kernel eigenvalue problem. Neural Com-
putation, 10(5):1299–1319, 1998.
[33] E. Shechtman and M. Irani. Matching local self-similarities
across images and videos. In CVPR, 2007.
[34] J. Shi and J. Malik. Normalized cuts and image segmenta-
tion. PAMI, 22(8):888–905, 2000.
[35] J. Sivic and A. Zisserman. Video Google: A text retrieval
approach to object matching in videos. In ICCV, 2003.
[36] R. Tavenard, H. J´ egou, and L. Amsaleg. Balancing clus-
ters to reduce response time variability in large scale image
search. In CBMI, 2011.
[37] G. Tolias, Y. Avrithis, and H. J´ egou. To aggregate or not
to aggregate: Selective match kernels for image search. In
ICCV, 2013.
[38] G. Tolias, T. Furon, and H. J´ egou. Orientation covariant ag-
gregation of local descriptors with embeddings. In ECCV,
2014.
[39] G. Tolias and H. J´ egou. Visual query expansion with or with-
out geometry: refining local descriptors by feature aggrega-
tion. Pattern Recognition, 47(10):3466–3476, 2014.
[40] G. Tolias, Y. Kalantidis, and Y. Avrithis. Symcity: Feature
selection by symmetry for large scale image retrieval. In
ACM Multimedia, 2012.
[41] A. Torii, J. Sivic, T. Pajdla, and M. Okutomi. Visual place
recognition with repetitive structures. In CVPR, 2013.
[42] P. Turcot and D. Lowe. Better matching with fewer features:
the selection of useful features in large database recognition
problems. In ICCV, 2009.
[43] A. Vedaldi and S. Soatto. Quick shift and kernel methods for
mode seeking. In ECCV, 2008.
[44] S. Winder and M. Brown. Learning local image descriptors.
In CVPR, 2007.
[45] S. Winder and G. Hua. Picking the best daisy. In CVPR,
2009.