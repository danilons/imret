# coding: utf-8
import click
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
                                "include('{tptp}').",
                                "include('{sumo}').",
                                "% SZS end BatchIncludes",
                                "% SZS start BatchProblems",
                                "{problems} {answers}",
                                "% SZS end BatchProblems"])

    def ontology_by_image(self, image):
        imgname = image
        imindex = self.images.index(image)
        frame = self.df[self.df['image'] == imgname]
        formulae = []
        for _ in xrange(1):
            formula = "(instance {} Image)".format(imgname).replace(".jpg", "")
            formulae.append(formula)

        objects = set(frame.object1) | set(frame.object2)
        for obj in objects:
            objname = "{}{}".format(obj, imindex)
            formula = "(instance {} {})".format(objname, obj.title())
            formulae.append(formula)

        for _, row in frame.iterrows():
            obj1 = "{}{}".format(row.object1, imindex)
            obj2 = "{}{}".format(row.object2, imindex)
            prep = row['preposition'].title().replace(" ", "")
            axiom_ = "(holdsDuring Now (orientation {} {} {}))".format(obj1, obj2, prep)
            formula = axiom_.replace("_", "")
            formulae.append(formula)

        return formulae

    def tptp_by_image(self, image, imindex=0):
        imgname = image
        frame = self.df[(self.df['image'] == imgname)]  # & (self.df['object1'] == object1) & (self.df['object2'] == object2)]
        objects = set(frame.object1) | set(frame.object2)
        axioms = []
        names = set()

        # for _ in xrange(1):
        #     axiom = "fof(kb_IRRC_{},axiom, (( s__instance(s__{}__m,s__Image)))).".format(self.index,
        #                                                                                  imgname.replace('.jpg', ''))
        #     axioms.append(axiom)
        #     self.index += 1

        for obj in objects:
            objname = "{}{}".format(obj, imindex)
            axiom = "fof(kb_IRRC_{},axiom,(( s__instance(s__{}__m,s__{}) ))).".format(self.index,
                                                                                      objname,
                                                                                      obj.title())
            axioms.append(axiom)
            self.index += 1
            if obj not in names:
                axiom = "fof(kb_IRRC_{},axiom,(( s__instance(s__{}, s__SetOrClass) ))).".format(self.index,
                                                                                                obj.title())
                axioms.append(axiom)
                self.index += 1
                names.add(obj)

        for _, row in frame.iterrows():
            obj1 = "{}{}".format(row.object1, imindex)
            obj2 = "{}{}".format(row.object2, imindex)
            prep = row['preposition'].title().replace(" ", "")
            axiom = "fof(kb_IRRC_{},axiom,(( s__orientation(s__{}__m,s__{}__m,s__{}) ))).".format(self.index,
                                                                                                  obj1,
                                                                                                  obj2,
                                                                                                  prep)
            axioms.append(axiom)
            self.index += 1

        return axioms

    def tptp_query(self, obj1, obj2, preposition):
        prep = preposition.title().replace("_", "")
        return """fof(conj1,conjecture, ( (? [V__X,V__Y] : (s__instance(V__X,s__{obj1}) & s__instance(V__Y,s__{obj2}) & s__orientation(V__X,V__Y,s__{prep}))) )).
               """.format(obj1=obj1.title(),
                          obj2=obj2.title(),
                          prep=prep)

    def prover(self, image, query, verbose=False):
        try:
            noun1, preposition, noun2 = query.split('-')
        except AttributeError:
            return False

        objects = set(self.df[self.df.image == image].object1) & set(self.df[self.df.image == image].object2)
        if noun1 not in objects or noun2 not in objects:
            return False

        index = self.images.index(image) + 1

        irrc_fp = tempfile.NamedTemporaryFile(delete=False)
        for axiom in self.tptp_by_image(image, index):  # noun1, noun2, index):
            irrc_fp.write(axiom + "\n")
            if verbose:
                print(axiom)
        irrc_fp.close()

        tptp_query = self.tptp_query(noun1, noun2, preposition)
        if verbose:
            print(tptp_query)
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
        if verbose:
            print(batch)

        cmd = "./{} {} {}".format(self.ltb_runner, batch_config_fp.name, self.eprover)
        if verbose:
            print(cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, shell=True)
        response, _ = process.communicate()
        proved = "Proof found!" in response
        if verbose:
            print(response)

        return proved

    def runquery(self, query, verbose=False):
        answers = []
        with click.progressbar(length=len(self.images), show_pos=True, show_percent=True) as bar:
            for nn, image in enumerate(self.images):
                try:
                    proved = self.prover(image=image, query=query, verbose=verbose)
                    answers.append((image, proved))
                except:
                    print("Something went wrong with image: {}, index: {}".format(image, nn))
                bar.update(1)
        return answers