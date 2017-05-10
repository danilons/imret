# coding: utf-8
import os
import re
import numpy as np
import subprocess


class KnowledgeBase:

    def __init__(self, df, ltb_runner, eprover, batch_config):
        self.ltb_runner = ltb_runner
        self.eprover = eprover
        self.batch_config = batch_config
        self.df = df
        self.images = list(set(df.image))
        self.regex = re.compile(b'\[\[(\w+)')

    def ontology_by_image(self, image):
        imgname = image
        imindex = self.images.index(image)
        frame = self.df[self.df['images'] == imgname]
        for _ in xrange(1):
            yield "(instance {} Image)".format(imgname).replace(".jpg", "")

        objects = set(frame.noun1) | set(frame.noun2)
        for obj in objects:
            objname = "{}{}".format(obj, imindex)
            yield "(instance {} {})".format(objname, obj.title())

        for obj in objects:
            objname = "{}{}".format(obj, imindex)
            yield "(contains {} {})".format(imgname.replace(".jpg", ""),
                                            objname)

        for _, row in frame.iterrows():
            obj1 = "{}{}".format(row.noun1, imindex)
            obj2 = "{}{}".format(row.noun2, imindex)
            yield "(orientation {} {} {})".format(obj1,
                                                  obj2,
                                                  row['predicted'].title()).replace("_", "")

    def tptp_by_image(self, image, position=0):
        for axiom in self.ontology_by_image(image):
            if axiom.startswith("(instance"):
                _, instance, classname = axiom.replace(")", "").split()
                yield "fof(kb_IRRC_{},axiom,(( s__instance(s__{}__m,s__{}) ))).".format(position,
                                                                                        instance,
                                                                                        classname)
            if axiom.startswith("(orientation"):
                _, obj1, obj2, relation = axiom.replace(")", "").split()
                yield "fof(kb_IRRC_{},axiom,(( s__orientation(s__{}__m,s__{}__m,s__{}) ))).".format(position, obj1, obj2, relation)
            position = position + 1

    def tptp_query(self, image, noun1, noun2, preposition):
        prep = preposition.title().replace("_", "")
        return """fof(conj1,conjecture, ( (? [V__X1,V__X2,V__X3] :
                    (s__instance(V__X1,s__Image) &
                     s__instance(V__X2,s__{noun1}) &
                     s__instance(V__X3,s__{noun2}) &
                     (s__holdsDuring(s__Now,'s__orientation(V__X2,V__X3,s__{prep})')) &
                     s__instance(s__{img}__m, s__Image))) )).
               """.format(noun1=noun1.title(), noun2=noun2.title(), prep=prep, img=image.replace(".jpg", ""))

    def prover(self, image, query):
        try:
            noun1, preposition, noun2 = query.split('-')
        except AttributeError:
            return image, None, []

        objects = set(self.df[self.df.image == image].object1) & set(self.df[self.df.image == image].object2)
        if noun1 not in objects or noun2 not in objects:
            return image, None, []
        
        import ipdb;
        ipdb.set_trace()

        tptp_query = self.tptp_query(image, noun1, noun2, preposition)
        with open('IRRC.tptp', 'w') as fp:
            for axiom in self.tptp_by_image(image):
                fp.write(axiom + "\n")

        with open('Problems.p', 'w') as fp:
            fp.write(tptp_query)

        with open("Answers.p", "w") as fp:
            fp.write("")

        cmd = '{} {} {}'.format(self.ltb_runner, self.batch_config, self.eprover)
        print("Runnning {}".format(cmd))
        with open(os.devnull, 'wb') as devnull:
            _ = subprocess.call([self.ltb_runner, self.batch_config, self.eprover],
                                 stdout=devnull, stderr=subprocess.STDOUT)

        response = []
        with open("Answers.p", "r") as fp:
            for line in fp.readlines():
                response.append(line.strip())

        m = self.regex.findall("\n".join(response))
        if m:
            return image, m[0], response
        return image, None, response

    def runquery(self, query):
        for image in self.images:
            image, imname, _ = self.prover(image=image, query=query)
            yield image, imname
            # return [(image, imname)]