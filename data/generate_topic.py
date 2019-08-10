import os

from summa import keywords


class GenerateTopic:
    def __init__(self,directory, parts):


        self.article_path = os.path.abspath(os.path.join(directory,parts)) + ".tsv"
        self.topic_path = self.article_path.split(".")[0] + "_topic"
        self.topic = None

    def topic2file(self):
        with open(self.topic_path, "w") as out:
            with open(self.article_path) as f:
                for line in f:
                    line_splt = line.split('\t')
                    out.write(keywords.keywords(line_splt[0]).replace("\n", " "))
                    out.write("\n")
                    #print(line_splt[1])

    def read_topic(self):
        topic = []
        with open(self.topic_path) as f:
            for line in f:
                topic.append(line)
        return topic

if __name__ == "__main__":
    topic_maker = GenerateTopic("/data/ct/abs_summarize/sumdata/transf_summ/dataset", "val")
    topic_maker.topic2file()
    print(len(topic_maker.read_topic()))
