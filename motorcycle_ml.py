from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import happybase


# Create Spark session with Hive support
spark = (
    SparkSession.builder
    .appName("Motorcycle_Value_Classification")
    .enableHiveSupport()
    .getOrCreate()
)

# Load data from Hive table
df = spark.sql("""
    SELECT
        name,
        selling_price,
        year,
        seller_type,
        owner,
        km_driven,
        ex_showroom_price
    FROM motorcycles
""")

# Drop rows with missing important values
df = df.na.drop()

# Create binary label:
# 1 = high value motorcycle
# 0 = standard value motorcycle
df = df.withColumn(
    "high_value",
    when(col("selling_price") >= 100000, 1).otherwise(0)
)

# Convert categorical columns into numeric indexes
seller_indexer = StringIndexer(
    inputCol="seller_type",
    outputCol="seller_type_index",
    handleInvalid="keep"
)

owner_indexer = StringIndexer(
    inputCol="owner",
    outputCol="owner_index",
    handleInvalid="keep"
)

df = seller_indexer.fit(df).transform(df)
df = owner_indexer.fit(df).transform(df)

# Assemble model features
assembler = VectorAssembler(
    inputCols=[
        "year",
        "km_driven",
        "ex_showroom_price",
        "seller_type_index",
        "owner_index"
    ],
    outputCol="features",
    handleInvalid="skip"
)

model_df = assembler.transform(df).select("features", "high_value")

# Split training and testing data
train_data, test_data = model_df.randomSplit([0.7, 0.3], seed=42)

# Train Random Forest classifier
rf = RandomForestClassifier(
    labelCol="high_value",
    featuresCol="features",
    numTrees=50,
    seed=42
)

rf_model = rf.fit(train_data)

# Make predictions
predictions = rf_model.transform(test_data)

# Evaluate model
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="high_value",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="high_value",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="high_value",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="high_value",
    predictionCol="prediction",
    metricName="f1"
)

accuracy = accuracy_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)

print("Motorcycle High-Value Classification Results")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Write metrics to HBase
data = [
    ("run_001", "metrics:accuracy", str(accuracy)),
    ("run_001", "metrics:precision", str(precision)),
    ("run_001", "metrics:recall", str(recall)),
    ("run_001", "metrics:f1", str(f1)),
]


def write_to_hbase_partition(partition):
    connection = happybase.Connection("master")
    connection.open()
    table = connection.table("motorcycle_metrics")

    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})

    connection.close()


rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

spark.stop()