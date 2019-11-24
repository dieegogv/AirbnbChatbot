import sqlite3
from airbnb_bot import read_corpus, db_connect, brain_dump, output_banner, train_bot, response, color
from config import confidence_req


def output_commands():
    print("""
    help|?           prints help
    quit|bye|exit    exit program
    braindump        dumps database
    brainsize        shows size of database
    trainbot         add new question and reply to database
    testbot          looks up response in database to question
    """)

print("Cargando airbnBot..")
sent_tokens, word_tokens = read_corpus()
connection, cursor = db_connect()
print("hecho.")
print("Presiona ? para ayura.")
while True:
    try:
        user_response = input(color.DARKCYAN + 'AIRBNBOT> ' + color.END).strip()
        if user_response in ['quit', 'bye', 'exit']:
            break
        elif user_response in ['?', 'help']:
            output_commands()
        elif user_response == 'braindump':
            brain_dump()
        elif user_response == 'brainsize':
            res = brain_dump(sizeonly=True)
            print(res)
        elif user_response == 'trainbot':
            res = train_bot(None, None)
            if res is not None:
                print(res)
        elif user_response == 'testbot':
            h = input("Question: ")
            h = h.lower()
            if h == '':
                continue
            res = response(h)
            if res is not None:
                resp, confidence, source = res
                print("AIRBNBOT Respuesta: " + resp + " (confidence: %s (%s))" % (confidence, source))
                if confidence < confidence_req:
                    print("AIRBNBOT: respondo demasiado lento, necesito más entrenamiento")
            else:
                print("AIRBNBOT Respuesta: Necesito más entrenamiento")
        else:
            print("Disculpa, no entendí escribe help o ? para ayuda")
    except KeyboardInterrupt:
        break
print("Sayonara..")