import java.io.IOException;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().replaceAll("[\\p{Punct}]", "").toLowerCase();
            StringTokenizer itr = new StringTokenizer(line);
            Set<String> uniqueWords = new HashSet<>();

            while (itr.hasMoreTokens()) {
                uniqueWords.add(itr.nextToken());
            }

            List<String> sortedWords = new ArrayList<>(uniqueWords);
            Collections.sort(sortedWords);

            for (int i = 0; i < sortedWords.size(); i++) {
                word.set(sortedWords.get(i));
                context.write(word, one);
                for (int j = i + 1; j < sortedWords.size(); j++) {
                    word.set(sortedWords.get(i) + ":" + sortedWords.get(j));
                    context.write(word, one);
                }
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        boolean success = job.waitForCompletion(true);
        if (!success) {
            System.exit(1);
        }

        // Configure and run the second MapReduce job
        Configuration confTwo = new Configuration();
        Job job2 = Job.getInstance(confTwo, "Compute Confidence");
        job2.setJarByClass(WordCount.class);
        job2.setMapperClass(ConfidenceMapper.class);
        job2.setReducerClass(ConfidenceReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(FloatWritable.class);
        FileInputFormat.addInputPath(job2, new Path(args[1])); // use the output of the first job as input
        FileOutputFormat.setOutputPath(job2, new Path(args[2])); // separate output path for second job

        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }

    public static class ConfidenceMapper extends Mapper<Object, Text, Text, Text> {
        private Text wordPair = new Text();
        private Text countAndTotal = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] wordCount = value.toString().split("\\t");
            if (wordCount[0].contains(":")) {
                String[] words = wordCount[0].split(":");
                context.write(new Text(words[0]), new Text("pair:" + wordCount[1]));
                context.write(new Text(words[1]), new Text("pair:" + wordCount[1]));
            } else {
                context.write(new Text(wordCount[0]), new Text("total:" + wordCount[1]));
            }
        }
    }

    public static class ConfidenceReducer extends Reducer<Text, Text, Text, FloatWritable> {
        private FloatWritable result = new FloatWritable();

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int total = 0;
            Map<String, Integer> pairCounts = new HashMap<>();

            for (Text val : values) {
                String[] parts = val.toString().split(":");
                if (parts[0].equals("total")) {
                    total = Integer.parseInt(parts[1]);
                } else {
                    pairCounts.put(parts[0], Integer.parseInt(parts[1]));
                }
            }

            for (Map.Entry<String, Integer> entry : pairCounts.entrySet()) {
                float confidence = (float) entry.getValue() / total;
                result.set(confidence);
                wordPair.set(key.toString() + "->" + entry.getKey());
                context.write(wordPair, result);
            }
        }
    }
}
