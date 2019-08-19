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
