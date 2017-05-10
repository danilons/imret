# coding: utf-8
import os
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
        self.regex = re.compile(b'\[\[(\w+)')
        self.batch = """
        % SZS start BatchConfiguration
        division.category LTB.SMO
        output.required Assurance
        output.desired Proof Answer
        limit.time.problem.wc 60
        % SZS end BatchConfiguration
        % SZS start BatchIncludes
        include('{sumo}').
        include('{tptp}').
        % SZS end BatchIncludes
        % SZS start BatchProblems
        include('{problem}').
        % SZS end BatchProblems"""

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

    def tptp_by_image(self, image, position=0):
        imgname = image
        imindex = self.images.index(image)
        frame = self.df[self.df['image'] == imgname]
        for _ in xrange(1):
            yield "(instance {} Image)".format(imgname).replace(".jpg", "")

        objects = set(frame.object1) | set(frame.object2)
        for obj in objects:
            objname = "{}{}".format(obj, imindex)
            yield "fof(kb_IRRC_{},axiom,(( s__instance(s__{}__m,s__{}) ))).".format(position,
                                                                                    objname,
                                                                                    obj.title())

        for _, row in frame.iterrows():
            obj1 = "{}{}".format(row.object1, imindex)
            obj2 = "{}{}".format(row.object2, imindex)
            prep = row['preposition'].title().replace(" ", "")
            yield "fof(kb_IRRC_{},axiom,(( (s__holdsDuring(s__Now,'s__orientation(s__{}__m,s__{}__m,s__{})')) ))).".format(position, obj1, obj2, prep)

    def tptp_query(self, image, noun1, noun2, preposition):
        prep = preposition.title().replace("_", "")
        return """fof(conj1,conjecture, ( (? [V__X1,V__X2,V__X3] :
                    (s__instance(V__X1,s__Image) &
                     s__instance(V__X2,s__{noun1}) &
                     s__instance(V__X3,s__{noun2}) &
                     (s__holdsDuring(s__Now,'s__orientation(V__X2,V__X3,s__{prep})')) &
                     s__instance(s__{img}__m, s__Image))) )).
               """.format(noun1=noun1.title(),
                          noun2=noun2.title(),
                          prep=prep, img=image.replace(".jpg", ""))

    def prover(self, image, query):
        try:
            noun1, preposition, noun2 = query.split('-')
        except AttributeError:
            return image, None, []

        objects = set(self.df[self.df.image == image].object1) & set(self.df[self.df.image == image].object2)
        if noun1 not in objects or noun2 not in objects:
            return image, None, []

        tptp_query = self.tptp_query(image, noun1, noun2, preposition)
        irrc_fp = tempfile.NamedTemporaryFile(delete=False)
        for axiom in self.tptp_by_image(image):
            irrc_fp.write(axiom + "\n")
        irrc_fp.close()

        problems_fp = tempfile.NamedTemporaryFile(delete=False)
        problems_fp.write(tptp_query)
        problems_fp.close()

        batch_config_fp = tempfile.NamedTemporaryFile(delete=False)
        batch = self.batch.format(sumo=self.sumo,
                                  tptp=irrc_fp.name,
                                  problem=problems_fp.name)
        batch_config_fp.write(batch)
        batch_config_fp.close()

        cmd = "./{} {} {}".format(self.ltb_runner, batch_config_fp.name, self.eprover)
        print('Running {cmd}'.format(cmd=cmd))

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, shell=True)
        response, _ = process.communicate()
        print(response)

        m = self.regex.findall("\n".join(response))
        if m:
            return image, m[0], response
        return image, None, response

    def runquery(self, query):
        for image in self.images:
            image, imname, _ = self.prover(image=image, query=query)
            yield image, imname
