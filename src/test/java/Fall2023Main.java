import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import com.codingame.gameengine.runner.MultiplayerGameRunner;
import com.codingame.gameengine.runner.simulate.GameResult;
import com.google.common.io.Files;

public class Fall2023Main {
    public static void main(String[] args) throws IOException, InterruptedException {
        int[] finalScores = getFinalScores();
        System.out.println(Arrays.toString(finalScores));
        writeScoresToFile(finalScores);
    }

    private static int[] getFinalScores() throws IOException, InterruptedException {
        MultiplayerGameRunner gameRunner = new MultiplayerGameRunner();
        gameRunner.setSeed(-8358938852454912011l);
        gameRunner.addAgent("python3 config/Boss.py", "TestBoss_1");
        gameRunner.addAgent("python3 config/starterAI.py", "TestBoss_2");
        gameRunner.setLeagueLevel(1);
        GameResult gameResult = gameRunner.simulate();

        int[] finalScores = new int[2];
        for (int i = 0; i < gameResult.scores.size(); i++) {
            finalScores[i] = gameResult.scores.get(i);
        }
        return finalScores;
    }

    
    private static void writeScoresToFile(int[] scores) throws IOException {
        FileWriter writer = new FileWriter("scores.txt");
        for (int i = 0; i < scores.length; i++) {
            writer.write(String.valueOf(scores[i]));
            writer.write("\n"); // 每個分數一行
        }
        writer.close();
    }

    private static String compile(String botFile) throws IOException, InterruptedException {

        File outFolder = Files.createTempDir();

        System.out.println("Compiling Boss.java... " + botFile);
        Process compileProcess = Runtime.getRuntime()
            .exec(new String[] { "bash", "-c", "javac " + botFile + " -d " + outFolder.getAbsolutePath() });
        compileProcess.waitFor();
        return "java -cp " + outFolder + " Player";
    }

    private static String[] compileTS(String botFile) throws IOException, InterruptedException {

        System.out.println("Compiling ... " + botFile);

        Process compileProcess = Runtime.getRuntime().exec(
            new String[] { "bash", "-c", "npx tsc --target ES2018 --inlineSourceMap --types ./typescript/readline/ "
                + botFile + " --outFile /tmp/Boss.js" }
        );
        compileProcess.waitFor();

        return new String[] { "bash", "-c", "node -r ./typescript/polyfill.js /tmp/Boss.js" };
    }
}
