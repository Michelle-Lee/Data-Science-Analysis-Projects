##############################################################################
##
## Converting hourly commidities data into weekly VWAP time series
## Weeks are determined by COT (Commitment of Traders) report dates
##

##############################################################################
## load libraries

library(dplyr)
library(lubridate)
library(xts)

##############################################################################
## setup environment and constants

# setwd('./Documents/COT Project')

price.data.dir  <- '.'
cot.data.dir    <- './COTData/data'
output.data.dir <- './Documents'

price.data.file <- 'CornData.csv'
cot.data.file.pat <- '^COT.+.txt$'
vwap.data.file    <- 'CornDF.csv'


##############################################################################
## read and wrangle data

# obtain filenames
price.file <- file.path(price.data.dir, price.data.file)

cot.files <-
  list.files(
    path=cot.data.dir,
    pattern=cot.data.file.pat,
    full.names=T
  )

# read price data and convert timestamps
price.df <-
  read.csv(price.file, stringsAsFactors=F) %>%
  mutate(bar_ts=ymd_hms(bar_ts, tz='US/Eastern')) %>%
  arrange(bar_ts)

# read COT data, filter corn data, and convert report dates
# to timestamps after appending standard US market reset time
cot.df <-
  lapply(
    cot.files,
    function(cot.file) {
      read.csv(cot.file, stringsAsFactors=F, strip.white=T) %>%
        filter(grepl('CORN', Market.and.Exchange.Names)) %>%
        mutate(
          Report.Date=
            ymd_hms(
              paste(As.of.Date.in.Form.YYYY.MM.DD, '18:00:00'),
              tz='US/Eastern'
            )
        )
    }
  ) %>%
  bind_rows() %>%
  arrange(Report.Date)

##############################################################################
## calculate weekly aggregate stats by COT report report dates

vwap.df <-
  price.df %>%
  mutate(
    Report.Date=
      # this one's tricky... we want to stuff a week's worth of data into a bin
      # that is labeled according to the COT report date which occurs at the
      # end of that week period (more specifically, at the US market reset time)
      cot.df$Report.Date[
        1 + cut(bar_ts, breaks=cot.df$Report.Date, labels=F, include.lowest=F, right=T)
        ]
  ) %>%
  group_by(Report.Date) %>%
  summarize(
    product=first(product),
    issue=first(issue),
    roll=sum(roll),
    vwap=sum(volume*close)/sum(volume),
    min_ts=min(bar_ts),
    max_ts=max(bar_ts)
  ) %>%
  # strip first and last partial bins
  head(-1) %>%
  tail(-1) %>%
  as.data.frame()


##############################################################################
## save vwap data

write.csv(
  vwap.df,
  file=file.path(vwap.data.file),
  row.names=F
)

##############################################################################
## plot potential target variables

my.vwap.xts <- vwap.xts
colnames(my.vwap.xts) <- 'ZC'
# plot(my.vwap.xts)

# my.vwap.xts$Diff.1W <- diff(my.vwap.xts$ZC)
# my.vwap.xts$Diff.2W <- diff(my.vwap.xts$ZC,2)
my.vwap.xts$Diff.4W <- diff(my.vwap.xts$ZC,4)
my.vwap.xts$Diff.6W <- diff(my.vwap.xts$ZC,6)
my.vwap.xts$Diff.8W <- diff(my.vwap.xts$ZC,8)
my.vwap.xts$PlotAdj <- 2*mean(abs(my.vwap.xts$Diff.6W), na.rm=T) * (my.vwap.xts$ZC - mean(my.vwap.xts$ZC)) / sd(my.vwap.xts$ZC)


plot(my.vwap.xts[,-1])

vwap.xts <- xts(vwap.df$vwap, date(vwap.df$Report.Date))
colnames(vwap.xts) <- 'ZC'

##############################################################################
## independent variables


# All categorical variables can be dropped since they only have one factor level
cot.df[,c('Market.and.Exchange.Names','As.of.Date.in.Form.YYMMDD','As.of.Date.in.Form.YYYY.MM.DD',
          'CFTC.Contract.Market.Code','CFTC.Market.Code.in.Initials','CFTC.Region.Code','CFTC.Commodity.Code',
          'Contract.Units','CFTC.Contract.Market.Code..Quotes.','CFTC.Market.Code.in.Initials..Quotes.',
          'CFTC.Commodity.Code..Quotes.')] <- list(NULL)

### Convert all of cot.df to numeric type before xts conversion.

cot.df <- data.frame(cot.df$Report.Date, sapply(cot.df[, names(cot.df) != "Report.Date"], as.numeric))
names(cot.df)[1] <- 'Report.Date'

# 0W diff
cot.xts <- xts(cot.df[, names(cot.df) != "Report.Date"], date(cot.df$Report.Date))
# 1W diff
cot.1w.xts <- as.data.frame(lapply(cot.xts,diff,1))
colnames(cot.1w.xts) <- c(sapply(colnames(cot.xts), function(x) paste0(x,'.1w.Diff')))
# 2W diff
cot.2w.xts <- as.data.frame(lapply(cot.xts,diff,2))
colnames(cot.2w.xts) <- c(sapply(colnames(cot.xts), function(x) paste0(x,'.2w.Diff')))
# 4W diff
cot.4w.xts <- as.data.frame(lapply(cot.xts,diff,4))
colnames(cot.4w.xts) <- c(sapply(colnames(cot.xts), function(x) paste0(x,'.4w.Diff')))
# 6W diff
cot.6w.xts <- as.data.frame(lapply(cot.xts,diff,6))
colnames(cot.6w.xts) <- c(sapply(colnames(cot.xts), function(x) paste0(x,'.6w.Diff')))
# 8w diff
cot.8w.xts <- as.data.frame(lapply(cot.xts,diff,8))
colnames(cot.8w.xts) <- c(sapply(colnames(cot.xts), function(x) paste0(x,'.8w.Diff')))
# 12w diff
cot.12w.xts <- as.data.frame(lapply(cot.xts,diff,12))
colnames(cot.12w.xts) <- c(sapply(colnames(cot.xts), function(x) paste0(x,'.12w.Diff')))
