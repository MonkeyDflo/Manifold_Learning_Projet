# packages ####
install.packages("dimRed")
install.packages("coRanking")
library(rgl) 
library(vegan)
library(plot3D)
library(rgl)
library(Rtsne)
library(dimRed)
library(coRanking)
# I. Datasets ####
## I.A. Datasets synthétiques ####
### 1) Anneaux concentriques lab 2 et lab 5 ####
anneaux <- function(n0 = 150){
  rayon_int <- runif(n0, 50,   80)  
  rayon_ext <- runif(n0, 200, 230)
  
  rayons    <- c(rayon_int, rayon_ext)
  angles    <- runif(2*n0, 0, 2*pi)
  
  return(cbind(rayons * cos(angles),
               rayons * sin(angles)))
}

n <- 200
x <- anneaux(n)
plot(x, pch = 19, col = rep(2:1, each = n))
### 2) 3-sensor dataset lab 3 ####
generateData <- function(n) {
  require(pdist)
  
  # these sensors where selected randomly
  sensors <- matrix(ncol = 3, data = 
                      c(0.026, 0.236, -0.653, 0.310, 0.507, -0.270, -0.466,  -0.140, 0.353, -0.473,
                        0.241, 0.193, 0.969, 0.094, 0.756, -0.978, -0.574, -0.502, -0.281, 0.993,
                        0.026, -0.913, -0.700, 0.876, 0.216, -0.739, 0.556, -0.155, 0.431, 0.411))
  
  # draw random points on the 3d unit cube
  unitcube <- matrix(runif(3 * n, -1, 1), ncol = 3)
  
  # We ode each point as the distance to sensors : intrinsic dimension = 3
  # while extrinsic dimension = 10
  X <- as.matrix(pdist(unitcube, sensors))
  noise <- matrix(rnorm(ncol(X) * nrow(X), sd = .01), ncol = ncol(X))
  return(X + noise)
}

res100   <- generateData(100)
res1000  <- generateData(1000)
res10000 <- generateData(10000)
# 
### 3) swiss roll ####
#library(rgl) 
#library(vegan)
#library(plot3D)
n <- 1000 # Random position on the parameteric domain.
u <- matrix(runif(2 * n), ncol = 2)
v <- 3 * pi / 2 * (0.1 + 2 * u[, 1])
x <- -cos(v) * v
y <- 20 * u[, 2]
z <- sin(v) * v
swissroll <- cbind(x, y , z)
plot3d(swissroll[order(v), ], col = rainbow(n), size = 10)
#http://www.sthda.com/english/wiki/impressive-package-for-3d-and-4d-graph-r-software-and-data-visualization
scatter3D(x,y,z, theta = 15, phi = 20, bty = "g",
          pch = 20, cex = 2, ticktype = "detailed")


### 4) sphere tentative flo -> Marche pas####
#library("rgl")
set.seed(101)
n <- 50
theta <- runif(n,0,2*pi)
u <- runif(n,-1,1)
x <- sqrt(1-u^2)*cos(theta)
y <- sqrt(1-u^2)*sin(theta)
z <- u
sphere = spheres3d(x,y,z,col="red",radius=0.02)
print(sphere)
### 5) helix ####
t = seq(0, 40*pi, by=pi/20)
#       VARIABLES
R=3;# MAJOR RADIUS
r=1;# MINOR RADIUS
n=6;# No. of loops

xt = (R + r*cos(n*t))*cos(t)
yt = r*sin(n*t)
zt = (R + r*cos(n*t))*sin(t)
scatter3D(xt,yt,zt, theta = 15, phi = 20, bty = "g",
          pch = 20, cex = 2, ticktype = "detailed")
helix <- cbind(xt, yt, zt)
### 6) helix in a torus ####
n = 10000
u = seq(0, 40*pi, by=pi/5)
v = u * 0.1 #number of loops in the torus
a=3
b=1
x=(a + (b * cos(u)) ) * cos(v) 
y=(a + (b * cos(u)) ) * sin(v) 
z= (b * sin(u)) 
#labels
print(length(z))
print(length(z[z<0.5]))

scatter3D(x,y,z, theta = 100, phi = 30, bty = "g",
          pch = 20, cex = 2, ticktype = "detailed")
scatter3D(x,y,z, theta = 100, phi = 100, bty = "g",
          pch = 20, cex = 2, ticktype = "detailed")
helix_in_torus <- cbind(x, y, z)
### 7) Sphere Thomas ####
#hypersphere
install.packages("BiocManager")
library(BiocManager)
BiocManager::install("graph")
install.packages("ggm")
library("ggm")
install.packages("rgl")
library(rgl)
n <- 1000
sphere <- rsphere(n,3)
#plot3d(sphere[order(sphere[,1]),],col = rainbow(n))
#plot3D(sphere[order(sphere[,1]),],col = rainbow(n))
#scatter3D(sphere[order(sphere[,1]),],col = rainbow(n))
#scatter3D(sphere,col = rainbow(n))
## I.B. Datasets IRL ####
### 1) Mnist #### 
setwd("~/Documents/Manifold Learning/Manifold_Projet/Manifold_Learning_Projet")
all    <- as.matrix(read.table("data.txt"))
labels <- read.table("labels.txt", colClasses = 'integer')
# II. Algos ####
## II.A. KACP (Cintia) ####
#ACP Classique
res.pca <- prcomp(swissroll, scale = TRUE)
fviz_eig(res.pca)
plot(res.pca$rotation)
plot(res.pca$sdev)
plot(res.pca$x) #ici on peut voir qu'on arrive a aplatir le swissroll
#Kernel ACP
k_acp <- kpca(as.matrix(swissroll), kernel="rbfdot", kpar = 
                list(sigma = 0.01), th = 1e-4, na.action = na.omit)
plot(eig(k_acp),  xlim = c(0,10))
xkpca_v <- kpca(swissroll, kernel = "vanilladot", kpar = list())
plot(pcv(xkpca_v), col = rainbow(nrow(swissroll)), pch = 10)
#mds classique
swissroll.mds <- cmdscale(dist(swissroll), k = 3, eig = FALSE, add = FALSE, x.ret = T)
## II.B. lle (Thomas) ####
library("lle")
# perform LLE
results <- lle( X=sphere, m=3, k=10, reg=2, ss=FALSE, id=TRUE, v=0.9 )
str( results )
# plot results and intrinsic dimension (manually)
plot( results$Y[order(sphere[,1]),],col=rainbow(n), main="embedded data", xlab=expression(y[1]), ylab=expression(y[2]) )
plot( results$id, main="intrinsic dimension", type="l", xlab=expression(x[i]), ylab="id", lwd=2 )
## II.C. t-SNE (Florent) ####
install.packages("Rtsne")
library(Rtsne)
### 1) tSNE sur IRIS ####
fit_iris <- Rtsne(valeurs,               # données
             pca = FALSE,           # initialisation
             perplexity = 30,       # paramètre à regler
             theta = 0.0)           # acceleration de l'algorithme
print(fit_iris)
plot(fit_iris$Y, col = etiquettes, pch = 19)

### 2) tSNE sur swissRoll ####
fit_swissroll <- Rtsne(swissroll,               # données
             pca = FALSE,           # initialisation
             perplexity = 30,       # paramètre à regler
             theta = 0.0)           # acceleration de l'algorithme
print(fit_swissroll)
plot(fit_swissroll$Y, col = rainbow(n), pch = 19)

### 3) tSNE sur helix ####
set.seed(1)
fit_helix <- Rtsne(helix,               # données
                            pca = FALSE,           # initialisation
                            perplexity = 30,       # paramètre à regler
                            theta = 0.0, # acceleration de l'algorithme
                   check_duplicates = FALSE)           
print(fit_helix)
plot(fit_helix$Y, col = rainbow(z), pch = 19)
### 4) tSNE sur helix in torus ####
fit_helix_in_torus <- Rtsne(helix_in_torus,               # données
                            pca = FALSE,           # initialisation
                            perplexity = 30,       # paramètre à regler
                            theta = 0.0,
             check_duplicates = FALSE)           # acceleration de l'algorithme
print(fit_helix_in_torus)
plot(fit_helix_in_torus$Y[order(fit_helix_in_torus$Y[,1])], col = rainbow(n), pch = 19)

### 5) t-SNE sur un échantillon de 1000 images de MNIST ####
fit_MNIST <- Rtsne(all,               # données
                            pca = FALSE,           # initialisation
                            perplexity = 30,       # paramètre à regler
                            theta = 0.0,
                            check_duplicates = FALSE) 
print(fit_MNIST)fl
plot(fit_MNIST$Y, color = rainbow(n), pch = 19)
### 6) t-SNE sphere ####
fit_sphere <- Rtsne(sphere,               # données
                   pca = FALSE,           # initialisation
                   perplexity = 30,       # paramètre à regler
                   theta = 0.0,
                   check_duplicates = FALSE) 
print(fit_sphere)
plot(fit_sphere$Y, col = rainbow(n), pch = 19)
# III. Comparaison méthodes de réduction de dimension ####
library(dimRed)
library(coRanking)
## III.A. Co-ranking matrix (Florent) ####
coranking(
  Xi, #high dimensional data
  X, #low dimensional data
  input_Xi = c("data", "dist", "rank"), #type of input of Xi (see. details)
  input_X = input_Xi, #type of input of X (see. details)
  use = "C"
)
### 1) coranking swissroll ####
#### i. kACP 
#### ii. lle 
#### iii. t-sne 
### 2) coranking helix ####
#### i. kACP 
#### ii. lle 
#### iii. t-sne
Q_helix = coranking(
  helix, #high dimensional data
  fit_helix$Y, #low dimensional data
  input_Xi = "data", #type of input of Xi (see. details)
  input_X = "data", #type of input of X (see. details)
  use = "C"
)
imageplot(
  Q_helix,
  lwd = 2,
  bty = "n",
  main = "co-ranking matrix tSNE helix",
  xlab = expression(R),
  ylab = expression(Ro),
  col = colorRampPalette(colors = c("gray85", "red", "yellow", "green", "blue"))(100),
  axes = FALSE,
  legend = TRUE,
)
#R_NX_helix = R_NX(Q_helix)

### 2bis) coranking helix_in_torus ####
#### i. kACP 
#### ii. lle 
#### iii. t-sne
Q_helix_in_torus = coranking(
  helix_in_torus, #high dimensional data
  fit_helix_in_torus$Y, #low dimensional data
  input_Xi = "data", #type of input of Xi (see. details)
  input_X = "data", #type of input of X (see. details)
  use = "C"
)
imageplot(
  Q_helix_in_torus,
  lwd = 2,
  bty = "n",
  main = "co-ranking matrix tSNE helix_in_torus",
  xlab = expression(R),
  ylab = expression(Ro),
  col = colorRampPalette(colors = c("gray85", "red", "yellow", "green", "blue"))(100),
  axes = FALSE,
  legend = TRUE,
)


### 3) coranking sphere ####
#### i. kACP 
#### ii. lle 
#### iii. t-sne
### 4) coranking Mnist ####
#### i. kACP 
#### ii. lle 
#### iii. t-sne
### test LCMC ####
#helix 
LCMC(Q_helix, K = 1:nrow(Q_helix))
#helix in torus
LCMC(Q_helix_in_torus, K = 1:nrow(Q_helix_in_torus))
# Rank Matrix -> pb 
rankmatrix_helix = rankmatrix(helix, input = "data", use = "C")
plot_R_NX(RN_X_helix, pal = grDevices::palette(), ylim = c(0, 0.9))
## III.B. Q_global et Q_local (Cintia) ####
## III.C. AUC + Cophenetic correlation (Thomas) ####
## III.D. Reconstruction error Bonus####


