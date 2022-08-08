from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/tinyroberta-squad2"

qa_pipeline = pipeline('question-answering',
                       model=model_name, tokenizer=model_name)

if __name__ == "__main__":
    qa_input = {
        'question': "Where is the State of Fun?",
        'context': "Sentosa - the State of Fun. With its pristine beaches, exciting attractions and tropical landscapes, the State of Fun is sure to leave you spellbound. A much-beloved venue, Resorts World™ Sentosa is home to a slew of enthralling experiences. Take a movie-themed ride at Universal Studios Singapore, or immerse yourself in aquatic adventures at the S.E.A. Aquarium™."
    }
    res = qa_pipeline(qa_input)
    print(res)
