from langchain_ai21.contextual_answers import (
    AI21ContextualAnswers,
    ContextualAnswerBody,
    AI21ContextualAnswersLLM,
)


context = """
Albert Einstein (/ˈaɪnstaɪn/ EYEN-styne;[4] German: [ˈalbɛɐt ˈʔaɪnʃtaɪn] ⓘ; 14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century.[1][5] His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation".[6] He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect",[7] a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science.[8][9] In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time.[10] His intellectual achievements and originality have made the word Einstein broadly synonymous with genius.[11]

Born in the German Empire, Einstein moved to Switzerland in 1895, forsaking his German citizenship (as a subject of the Kingdom of Württemberg)[note 1] the following year. In 1897, at the age of seventeen, he enrolled in the mathematics and physics teaching diploma program at the Swiss Federal polytechnic school in Zürich, graduating in 1900. In 1901, he acquired Swiss citizenship, which he kept for the rest of his life. In 1903, he secured a permanent position at the Swiss Patent Office in Bern. In 1905, he submitted a successful PhD dissertation to the University of Zurich. In 1914, he moved to Berlin in order to join the Prussian Academy of Sciences and the Humboldt University of Berlin. In 1917, he became director of the Kaiser Wilhelm Institute for Physics; he also became a German citizen again, this time as a subject of the Kingdom of Prussia.[note 1]

In 1933, while he was visiting the United States, Adolf Hitler came to power in Germany. Horrified by the Nazi "war of extermination" against his fellow Jews,[12] Einstein decided to remain in the US, and was granted American citizenship in 1940.[13] On the eve of World War II, he endorsed a letter to President Franklin D. Roosevelt alerting him to the potential German nuclear weapons program and recommending that the US begin similar research. Einstein supported the Allies but generally viewed the idea of nuclear weapons with great dismay.[14]

In 1905, sometimes described as his annus mirabilis (miracle year), Einstein published four groundbreaking papers.[15] These outlined a theory of the photoelectric effect, explained Brownian motion, introduced his special theory of relativity—a theory which addressed the inability of classical mechanics to account satisfactorily for the behavior of the electromagnetic field—and demonstrated that if the special theory is correct, mass and energy are equivalent to each other. In 1915, he proposed a general theory of relativity that extended his system of mechanics to incorporate gravitation. A cosmological paper that he published the following year laid out the implications of general relativity for the modeling of the structure and evolution of the universe as a whole.[16][17] The middle part of his career also saw him making important contributions to statistical mechanics and quantum theory. Especially notable was his work on the quantum physics of radiation, in which light consists of particles, subsequently called photons. For much of the last phase of his academic life, Einstein worked on two endeavors that proved ultimately unsuccessful. Firstly, he advocated against quantum theory's introduction of fundamental randomness into science's picture of the world, objecting that "God does not play dice".[18] Secondly, he attempted to devise a unified field theory by generalizing his geometric theory of gravitation to include electromagnetism too. As a result, he became increasingly isolated from the mainstream of modern physics.
"""


_GOOD_QUESTION = "When did Albert Einstein born?"
_BAD_QUESTION = "Who was John Hancock?"


def test_invoke():
    llm = AI21ContextualAnswersLLM()
    llm = AI21ContextualAnswers()
    prompt = ContextualAnswerBody(context=context, question=_BAD_QUESTION)
    response = llm.invoke(prompt)

    print(response)
