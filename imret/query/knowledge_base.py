# coding: utf-8
import re
import tempfile
import subprocess


class KnowledgeBase:

    def __init__(self, df, ltb_runner, eprover, sumo):
        self.ltb_runner = ltb_runner
        self.eprover = eprover
        self.sumo = sumo
        self.df = df
        self.images = list(set(df.image))
        self.index = 0
        self.batch = "\n".join(["% SZS start BatchConfiguration",
                                "division.category LTB.SMO",
                                "output.required Assurance",
                                "output.desired Proof Answer",
                                "limit.time.problem.wc 60",
                                "% SZS end BatchConfiguration",
                                "% SZS start BatchIncludes",
                                "include('{sumo}').",
                                "include('{tptp}').",
                                "% SZS end BatchIncludes",
                                "% SZS start BatchProblems",
                                "{problems} {answers}",
                                "% SZS end BatchProblems"])

    def ontology_by_image(self, image):
        imgname = image
        imindex = self.images.index(image)
        frame = self.df[self.df['image'] == imgname]
        for _ in xrange(1):
            yield "(instance {} Image)".format(imgname).replace(".jpg", "")

        objects = set(frame.object1) | set(frame.object2)
        for obj in objects:
            objname = "{}{}".format(obj, imindex)
            yield "(instance {} {})".format(objname, obj.title())

        for _, row in frame.iterrows():
            obj1 = "{}{}".format(row.object1, imindex)
            obj2 = "{}{}".format(row.object2, imindex)
            prep = row['preposition'].title().replace(" ", "")
            axiom_ = "(holdsDuring Now (orientation {} {} {}))".format(obj1, obj2, prep)
            yield axiom_.replace("_", "")

    def tptp_by_image(self, image, imindex=0):
        imgname = image
        frame = self.df[self.df['image'] == imgname]

        objects = set(frame.object1) | set(frame.object2)
        for obj in objects:
            objname = "{}{}".format(obj, imindex)
            yield "fof(kb_IRRC_{},axiom,(( s__instance(s__{}__m,s__{}) ))).".format(self.index,
                                                                                    objname,
                                                                                    obj.title())
            self.index += 1

        for _, row in frame.iterrows():
            obj1 = "{}{}".format(row.object1, imindex)
            obj2 = "{}{}".format(row.object2, imindex)
            prep = row['preposition'].title().replace(" ", "")
            yield "fof(kb_IRRC_{},axiom,(( s__orientation(s__{}__m,s__{}__m,s__{})))).".format(self.index, obj1, obj2, prep)
            self.index += 1

    def tptp_query(self, obj1, obj2, preposition, index=0):
        obj1_ = "{}{}".format(obj1, index)
        obj2_ = "{}{}".format(obj2, index)
        # prep = preposition.title().replace("_", "")
        # return """fof(conj1, conjecture, s__orientation(s__{obj1},s__{obj2},s__{prep})).
        #        """.format(obj1=obj1.title(),
        #                   obj2=obj2.title(),
        #                   prep=prep)
        prep = preposition.title().replace("_", "")
        # return "fof(conj1, conjecture, ((? [V__X, V__Y]: s__orientation(V__X, V__Y, s__{})) )).".format(prep.title())
        return "fof(conj1,conjecture, ( s__orientation(s__{obj1}__m,s__{obj2}__m,s__{prep}) )).".format(obj1=obj1_,
                                                                                                        obj2=obj2_,
                                                                                                        prep=prep)

    def prover(self, image, query):
        try:
            noun1, preposition, noun2 = query.split('-')
        except AttributeError:
            return image, None, []

        objects = set(self.df[self.df.image == image].object1) & set(self.df[self.df.image == image].object2)
        if noun1 not in objects or noun2 not in objects:
            return image, None, []
        index = self.images.index(image)

        irrc_fp = tempfile.NamedTemporaryFile(delete=False)
        for axiom in self.tptp_by_image(image, index):
            irrc_fp.write(axiom + "\n")
        irrc_fp.close()

        tptp_query = self.tptp_query(noun1, noun2, preposition, index)
        problems_fp = tempfile.NamedTemporaryFile(delete=False)
        problems_fp.write(tptp_query)
        problems_fp.close()

        answers_fp = tempfile.NamedTemporaryFile(delete=False)
        answers_fp.write("")
        answers_fp.close()

        batch_config_fp = tempfile.NamedTemporaryFile(delete=False)
        batch = self.batch.format(sumo=self.sumo,
                                  tptp=irrc_fp.name,
                                  problems=problems_fp.name,
                                  answers=answers_fp.name)
        batch_config_fp.write(batch)
        batch_config_fp.close()

        cmd = "./{} {} {}".format(self.ltb_runner, batch_config_fp.name, self.eprover)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, shell=True)
        response, _ = process.communicate()
        proved = "Proof found!" in response
        return image, proved, response

    def runquery(self, query):
        for image in self.images:
            image, imname, _ = self.prover(image=image, query=query)
            yield image, imname
