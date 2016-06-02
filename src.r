path <- "../" # where the rgf1.2 folder is 
# RGFCV is parallelized with foreach 
library(readr)
library(Metrics)
library(cvTools)
library(foreach)
library(doParallel)
library(ggplot2)
library(reshape2)
library(zoo)

# norund=(integer)*100 
# clusterNum=integer (using for RGFCV)
RGF <- function(train,test,target,nround,lambda=1,clusterNum=""){
path <- path 
train.x <- train
train.y <- target
test.x <- test 
# save the data
print("writing data")
write.table(train.x,paste0(path,"rgf1.2/test/sample/train",clusterNum,".data.x"), col.names=F,row.names=F,sep=" ")
write.table(train.y,paste0(path,"rgf1.2/test/sample/train",clusterNum,".data.y"), col.names=F,row.names=F,sep=" ")
write.table(test.x,paste0(path, "rgf1.2/test/sample/test",clusterNum,".data.x"), col.names=F,row.names=F,sep=" ")

# train.inp
pars <- paste0(
"@reg_L2=",lambda,",model_fn_prefix=",path,"rgf1.2/test/output/regress",clusterNum,".lam",lambda,".model","\n",
"train_x_fn=",path,"rgf1.2/test/sample/train",clusterNum,".data.x","\n",
"train_y_fn=",path,"rgf1.2/test/sample/train",clusterNum,".data.y","\n",
"algorithm=RGF","\n",
"loss=LS","\n",
"test_interval=100","\n", # 500
"max_leaf_forest=",nround,"\n", # 30000
"Verbose","\n",
"NormalizeTarget","\n"
#,"model_fn_for_warmstart=",path,"output/regress.lam0.001CV",NUMCV,".model-60"
)
writeLines(pars,paste0(path,"rgf1.2/test/sample/train",clusterNum,".inp"))

# start training
system(paste0("perl ",
 paste0(path,"rgf1.2/test/call_exe.pl "),
 paste0(path,"rgf1.2/bin/rgf "), "train ",
 paste0(path,"rgf1.2/test/sample/train",clusterNum)))
 
# predict.inp
N <- nround / 100  

NUM <- N 
if(NUM<10){NUM <- paste0("0",NUM)}
pars <- paste0("test_x_fn=",path,"rgf1.2/test/sample/test",clusterNum,".data.x","\n",
"model_fn=",path,"rgf1.2/test/output/regress",clusterNum,".lam",lambda,".model-",NUM,"\n",
"prediction_fn=",path,"rgf1.2/test/output/regress",clusterNum,".lam",lambda,".pred-",NUM,"\n")
writeLines(pars,paste0(path,"rgf1.2/test/sample/predict",clusterNum,".inp"))
# predicting 
system(paste0("perl ",
 paste0(path,"rgf1.2/test/call_exe.pl "),
 paste0(path,"rgf1.2/bin/rgf "), "predict ",
 paste0(path,"rgf1.2/test/sample/predict",clusterNum)))
pred <- read_csv(paste0(path,"rgf1.2/test/output/regress",clusterNum,".lam",lambda,".pred-",NUM),col_names=F)
return(list(prediction=pred$X1))
}

RGFCV <- function(train,target,nround=10000,lambda=0.01,min_pop=10, Seed=131, nfold=5){
path <- path 
cv.folds <- cvFolds(nrow(train),nfold)
cvs <- cv.folds$which

gc()
cl <- makePSOCKcluster(nfold)
registerDoParallel(cl)
# making training model
TMP <- foreach(NUMCV=1:nfold,.packages=c("readr","Metrics"))%dopar%{
test.x <- train[which(cvs==NUMCV),]
test.y <- target[which(cvs==NUMCV)]
train.x <- train[which(cvs!=NUMCV),]
train.y <- target[which(cvs!=NUMCV)]

print("writing data")
write.table(train.x,paste0(path,"rgf1.2/test/sample/trainCV",NUMCV,".data.x"), col.names=F,row.names=F,sep=" ")
write.table(train.y,paste0(path,"rgf1.2/test/sample/trainCV",NUMCV,".data.y"), col.names=F,row.names=F,sep=" ")
write.table(test.x,paste0(path, "rgf1.2/test/sample/testCV",NUMCV,".data.x"), col.names=F,row.names=F,sep=" ")
write.table(test.y,paste0(path, "rgf1.2/test/sample/testCV",NUMCV,".data.y"), col.names=F,row.names=F,sep=" ")

pars <- paste0(
"@reg_L2=",lambda,",model_fn_prefix=",path,"rgf1.2/test/output/regress.lam",lambda,"CV",NUMCV,".model","\n",
"train_x_fn=",path,"rgf1.2/test/sample/trainCV",NUMCV,".data.x","\n",
"train_y_fn=",path,"rgf1.2/test/sample/trainCV",NUMCV,".data.y","\n",
"algorithm=RGF","\n",
"loss=LS","\n",
"test_interval=100","\n", # 500
"max_leaf_forest=",nround,"\n", # 50000
"min_pop=",min_pop,"\n",
"Verbose","\n",
"NormalizeTarget","\n")
writeLines(pars,paste0(path,"rgf1.2/test/sample/train",NUMCV,".inp"))

system(paste0("perl ",
 paste0(path,"rgf1.2/test/call_exe.pl "),
 paste0(path,"rgf1.2/bin/rgf "), "train ",
 paste0(path,"rgf1.2/test/sample/train",NUMCV)))
 
# prediction 
print("read data")
# predict.inp
N <- nround / 100  
score <- rep(0,N)
for(i in 1:N){
NUM <- i 
if(NUM<10){NUM <- paste0("0",NUM)}
pars <- paste0("test_x_fn=",path,"rgf1.2/test/sample/testCV",NUMCV,".data.x","\n",
"model_fn=",path,"rgf1.2/test/output/regress.lam",lambda,"CV",NUMCV,".model-",NUM,"\n",
"prediction_fn=",path,"rgf1.2/test/output/regress.lam",lambda,"CV",NUMCV,".pred-",NUM,"\n")
writeLines(pars,paste0(path,"rgf1.2/test/sample/predict",NUMCV,".inp"))
# start predicting 
system(paste0("perl ",
 paste0(path,"rgf1.2/test/call_exe.pl "),
 paste0(path,"rgf1.2/bin/rgf "), "predict ",
 paste0(path,"rgf1.2/test/sample/predict",NUMCV)))
pred <- read_csv(paste0(path,"rgf1.2/test/output/regress.lam",lambda,"CV",NUMCV,".pred-",NUM),col_names=F)
score[i] <- logLoss(test.y,pred$X1)
}
score 
}
stopCluster(cl)

N <- nround / 100  
score <- rep(0,N)
tmp <- rep(0,N)
for(j in 1:nfold){ tmp <- tmp + TMP[[j]]}
score <- tmp / nfold 

# plot 
tmp <- data.frame(ntree=1:N,score)
tmp <- melt(tmp, id="ntree", measure=c("score"))
g <- ggplot(tmp, aes(x=ntree, y=value, colour=variable, group=variable)) + geom_line() 
g <- g + ggtitle("RGF_logLoss")
plot(g)
return(list("bestNum"=which.min(score),"best_logLoss"=min(score)))
}
