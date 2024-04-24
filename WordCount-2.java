import java.io.*;
import java.net.*;
import java.util.*;
import org.apache.hadoop.*;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class WordCount
{

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>
    {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
	private Text wordj = new Text();

	private Configuration conf;
private BufferedReader fis;


        public void map(Object key, Text value, Context context
                        ) throws IOException, InterruptedException 
        {

String line = value.toString();


            // 𝐴 new ArrayList
            ArrayList<String> A = new ArrayList<String>();
            StringTokenizer itr = new StringTokenizer(line);
            // for each word w∈ 𝑙𝑙 //this loop finds a list of unique words in a line
            while (itr.hasMoreTokens()) 
            {
		String newWord = new String(itr.nextToken());
                // if(A.contains(w) = false)
               // word.set(itr.nextToken());
                if (A.contains(newWord) == false)
                	A.add(newWord);
            }
	

	/*    String[] wordsArray = line.split("\\s+");
	    Set<String> uniqueWords = new HashSet<>(Arrays.asList(wordsArray));
	    ArrayList<String> A = new ArrayList<>(uniqueWords);
          */  // SORT(𝐴𝐴);
            Collections.sort(A);
            // for each word 𝑤𝑤𝑖𝑖 ∈ 𝐴𝐴, 0 ≤ 𝑖𝑖 ≤ 𝐴𝐴. 𝑙𝑙𝑙𝑙𝑙𝑙𝑙𝑙𝑙𝑙ℎ() − 1
            for(int i = 0; i < A.size(); i++)
            {
                // EMIT (word 𝑤𝑤𝑖𝑖, one);
                word = new Text(A.get(i));
                context.write(word, one);
                // for each word 𝑤𝑤𝑗𝑗 ∈ 𝐴𝐴, 𝑖𝑖 < 𝑗𝑗 ≤ 𝐴𝐴. 𝑙𝑙𝑙𝑙𝑙𝑙𝑙𝑙𝑙𝑙ℎ()-1
                for (int j = i + 1; j < A.size(); j++)
                {
                    // EMIT (word pair {𝑤𝑤𝑖𝑖: 𝑤𝑤𝑗𝑗}, one)
                    word = new Text(A.get(i)+ ":" + A.get(j));
                    context.write(word, one);
                }
            }
	//A.clear();
        }
    }
    public static class IntSumReducer extends Reducer<Text,IntWritable,Text,IntWritable> 
    {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                        Context context
                        ) throws IOException, InterruptedException 
        {
            int sum = 0;
            for (IntWritable val : values) 
            {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception 
    {
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
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

  
