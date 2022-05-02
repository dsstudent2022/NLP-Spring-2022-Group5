Run each of the models (NB, SVM, LSTM) one at a time with repeating the pre-processing code for each.

Word Clouds were made in R by using:

rm(list = ls())
setwd("~/Desktop/Natural_LP")

#==================================================================================================
install.packages("webshot")
install.packages("Hmisc")
install.packages("quanteda")
Yes
install.packages("tidytext")
install.packages("SnowballC")
install.packages("textstem")
Yes
install.packages("wordcloud2")
install.packages("readtext")

library(tidyverse)
library(dplyr)
library(Hmisc)
library(data.table)
library(lubridate)
library(quanteda)
library(tidytext)
library(SnowballC)
library(textstem)
library(wordcloud2)
library(htmlwidgets)
library(webshot)
webshot::install_phantomjs(force = TRUE)
library(webshot)
library(readtext)
#==================================================================================================
dat <- read.csv('GFC_keywords.csv')
names(dat)[2] <- "Keyword"
names(dat)[3] <- "Significance"
df_00_10 = subset(dat, select = c(2,3) )

cloud_00_10 <- wordcloud2(data = df_00_10, size = 1.6, color = 'random-dark')
saveWidget(cloud_00_10, "tmp.html", selfcontained = F)
webshot("tmp.html", paste0("./word_clouds/unigram_wordcloud_GFC",".png"),
        delay = 15, vwidth = 2000, vheight = 2000)