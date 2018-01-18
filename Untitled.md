

```python
import os 
import sys

spark_home = os.environ.get('SPARK_HOME', None)
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.10.4-src.zip'))
sys.path.insert(0, os.path.join(spark_home, 'python'))
execfile(os.path.join(os.environ["SPARK_HOME"], 'python/pyspark/shell.py'))
```

    Welcome to
          ____              __
         / __/__  ___ _____/ /__
        _\ \/ _ \/ _ `/ __/  '_/
       /__ / .__/\_,_/_/ /_/\_\   version 2.2.1
          /_/
    
    Using Python version 2.7.13 (default, Dec 20 2016 23:05:08)
    SparkSession available as 'spark'.



```python

```
