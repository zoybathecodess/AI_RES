import pandas as pd
from typing import Iterator, Tuple




def rolling_time_splits(dates: pd.Series, initial_train_days:int, horizon_days:int, step_days:int) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
"""Generate rolling (train_idx, test_idx) index sets based on dates series (datetime64).


- dates: pandas Series of datetimes (index aligned with rows)
- initial_train_days, horizon_days, step_days: ints interpreted as days


Yields tuples of (train_index, test_index) as pandas DTI or integer arrays.
"""
if not pd.api.types.is_datetime64_any_dtype(dates):
raise ValueError("dates must be datetime dtype")


start = dates.min()
end = dates.max()
train_start = start
train_end = train_start + pd.Timedelta(days=initial_train_days)


while True:
test_start = train_end
test_end = test_start + pd.Timedelta(days=horizon_days)
if test_start >= end:
break
# select indices
train_idx = dates[dates < train_end].index
test_idx = dates[(dates >= test_start) & (dates < test_end)].index
if len(test_idx) == 0:
break
yield (train_idx, test_idx)
# roll
train_end = train_end + pd.Timedelta(days=step_days)




# Simple integer-based generator (if users prefer by-row splits)


def rolling_integer_splits(n_rows:int, initial_train:int, horizon:int, step:int):
start = initial_train
while start + horizon <= n_rows:
train_idx = list(range(0, start))
test_idx = list(range(start, start + horizon))
yield (train_idx, test_idx)
start += step
