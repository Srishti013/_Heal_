import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tensorflow
import random
import json
import tflearn
import pickle as pickle
import nltk

data= {"intents": [
        {"tag": "greeting",
         "patterns": ["Hi",
                      "I need to talk to someone!",
                      "Is anyone there?",
                      "Hello",                  
                      "I need help",
                      "I cant take it anymore",
                      "I need someone to listen to me",
                      "Hey",
                      "How are you doing",
                      "Is anyone there?",
                      "What's up?",
                      "Good day",
                      "What's popping",
                      "hey"],
         "responses": ["Hello! How are you doing today?",
                       "Hey! Welcome to HEAL.",
                       "Hi there, I'm doing well! My name's HEAL, what can I do for you?",
                       "Hey there, my name's HEAL, your mental health friend. How can I help you today?",
                       "Hello, this is HEAL, your beloved Therapist. Come on, let's talk!",
                       "Good to see you! Let's begin today's session",
                       "Hi there, how can I help?",
                       "Is there something you want to talk about? HEAL is here for you!",
                       "Do you want to talk about something?",
                       "I am here to help you, we can talk about your feelings",
                       "Hey I am HEAL, your very own Therapist! Feel free to talk about anything",
                       "Hi its HEAL, your personal Therapist! Let's talk :)"],
         "context_set": ""
        },
        {
            "tag" : "salutation",
         "patterns": ["how are you?", "how was your day"],
         "responses": ["I'm great and more than happy to help!"]
        },
        {"tag": "goodbye",
         "patterns": ["Goodbye! This was helpful!",
                      "Bye",
                      "byebye",
                      "I have to go",
                      "I should go thanks for talking."
                      "Bye", "See you later!", "Goodbye!", "Thank you","Good day!"],
         "responses": ["Goodbye, I hope you will take better care of yourself. Looking forward to our next session.",
                       " I Hope HEAL the Therapist was helpful to you! Let's talk soon!",
                       "See you later, don't forget what we talked about today! I hope you would feel better after talking to me!",
                       "I hope you feel better after talking! Take care of yourself, and remember to check in soon!",
                       "Goodbye, Hope you have a good day ahead :)"
                       "See you later! Take care!",
                       "Have a lovely day! Take care and stay safe!",
                       "Take care of yourself! I'm always here to support you! Feel free to come back at any time!"],
         "context_set": ""
        },
        
      {
          "tag": "Mental illness",
          "patterns": ["What does it mean to have a mental illness?"],
          "responses": ["Mental illnesses are health conditions that disrupt a personls thoughts, emotions, relationships, and daily functioning. They are associated with distress and diminished capacity to engage in the ordinary activities of daily life. Mental illnesses fall along a continuum of severity: some are fairly mild and only interfere with some aspects of life, such as certain phobias. On the other end of the spectrum lie serious mental illnesses, which result in major functional impairment and interference with daily life. These include such disorders as major depression, schizophrenia, and bipolar disorder, and may require that the person receives care in a hospital. It is important to know that mental illnesses are medical conditions that have nothing to do with a personls character, intelligence, or willpower. Just as diabetes is a disorder of the pancreas, mental illness is a medical condition due to the brains biology. Similarly to how one would treat diabetes with medication and insulin, mental illness is treatable with a combination of medication and social support. These treatments are highly effective, with 70-90 percent of individuals receiving treatment experiencing a reduction in symptoms and an improved quality of life. With the proper treatment, it is very possible for a person with mental illness to be independent and successful."
              
          ],
      },
      {
          "tag": "effect",
          "patterns": ["Who does mental illness affect?"],
          "responses": ["It is estimated that mental illness affects 1 in 5 adults in America, and that 1 in 24 adults have a serious mental illness. Mental illness does not discriminate; it can affect anyone, regardless of gender, age, income, social status, ethnicity, religion, sexual orientation, or background. Although mental illness can affect anyone, certain conditions may be more common in different populations. For instance, eating disorders tend to occur more often in females, while disorders such as attention deficit/hyperactivity disorder is more prevalent in children. Additionally, all ages are susceptible, but the young and the old are especially vulnerable. Mental illnesses usually strike individuals in the prime of their lives, with 75 percent of mental health conditions developing by the age of 24. This makes identification and treatment of mental disorders particularly difficult, because the normal personality and behavioral changes of adolescence may mask symptoms of a mental health condition. Parents and caretakers should be aware of this fact, and take notice of changes in their childÃ¢â‚¬â„¢s mood, personality, personal habits, and social withdrawal. When these occur in children under 18, they are referred to as serious emotional disturbances (SEDs)."
              
          ],
      },
      {
          "tag": "Recover",
          "patterns": ["Can people with mental illness recover?"],
          "responses": ["When healing from mental illness, early identification and treatment are of vital importance. Based on the nature of the illness, there are a range of effective treatments available. For any type of treatment, it is essential that the person affected is proactive and fully engaged in their own recovery process. Many people with mental illnesses who are diagnosed and treated respond well, although some might experience a return of symptoms. Even in such cases, with careful monitoring and management of the disorder, it is still quite possible to live a fulfilled and productive life."
              
          ],
      },
      {
          "tag": "Fine",
          "patterns": ["Nothing much",
                       "I'm fine",
                       "I'm good",
                       "I'm okay"
                       ],
          "responses": ["How can I help you today?"
              
          ],
      },
      {
          "tag": "Find therapy",
          "patterns": ["Where can I go to find therapy?"],
          "responses": ["Different kinds of therapy are more effective based on the nature of the mental health condition and/or symptoms and the person who has them (for example, children will benefit from a therapist who specializes in childrens mental health). However, there are several different types of treatment and therapy that can help."
          ],
      },
      {
          "tag": "thanks",
          "patterns": ["Thanks", "Thank you", "That's helpful", "Thank's a lot!"],
          "responses": ["Happy to help!", "Any time!", "My pleasure"]
      },
      {
          "tag": "signs",
          "patterns": [
              "What are the signs i have bad mental health",
              "signs that could indicate bad mental heath",
              "What are some of the warning signs of mental illness?"
              
          ],
       "responses": [
           "Various signs of a bad mental health are 1.Feeling anxious or worried 2. Feeling anxious or worried 3. Emotional outburst 4. Sleep problem 4. Appetite changes 5.Quiet and withdrawn"
           
       ]
      },
      {
          "tag": "positive",
          "patterns": [
              "provide tips and guidance",
              "How to improve my mental health?",
              "How can i be positive "
          ],
       "responses": [
           "Talking – By talking about your thoughts and feelings, people will be able to help you problem solve and get the best support. Connecting – Connecting with friends and family on a regular basis can help boost mood and reduce stress. Values – Staying connected to your values and things that are important to you will help provide you with a sense of meaning. Thinking errors – Writing down your thoughts on a piece of paper and identifying thinking errors will help you spot any unhelpful thinking styles. Relaxation – Scheduling in time for yourself where you can relax is important for your overall wellbeing. Exercise – Engaging in regular exercise can support you to feel better and less stressed. Mindfulness – Engaging in mindfulness activities can calm your mind and body and help you stay focused on the present moment. Strengths – Identifying and working towards your strengths daily will provide you with an extra sense of confidence. Problem solve – Problem solving can help reduce worry and provide you with a practical solution to any problems you are facing. Challenging thoughts – Challenging unhelpful thoughts can be a useful technique for helping to provide you with accurate, realistic and more helpful thoughts." 
       ]
      },
      {
          "tag": "about",
          "patterns": [
              "What do you do?",
              "Who are you?",
              "What are you here for?",
              "What can you help me with?",
              "Is anyone there?",
              "Are you a real person?",
              "Tell me about yourself!"
          ],
          "responses": [
              "Love that question! My name's HEAL, your mental health friend! I've been trained to support you through any of the issues and things that you're going through in life right now. I'm all ears and want to support you through your ups and downs. Technically, I'm a computer that's been trained by a human, but I like to think of myself as human!",
              "Hey! I love that question! My name is HEAL, and I want to be your mental health friend and support you! My fellow human friends have trained me to be a compassionate listener and support buddy when things are going well, and particularily when things aren't going so well."
          ]
      },
      {
          "tag": "anxiety",
          "patterns": [
              "I think I have anxiety",
              "What is anxiety?",
              "anxiety",
              "Tell me about anxiety",
              "I want to know about anxiety",
              "Can I have some information about anxiety?",
              "How do I support a loved one with anxiety?",
              "How do I fix my anxiety?"
          ],
          "responses": [
              "I see that you want to learn more about anxiety, and how to support yourself or your loved ones. At the higest level, anxiety is your body's natural response to stress. It's a feeling of fear or apprehension about what's to come. The first day of school, going to a job interview, or giving a speech may cause most people to feel fearful and nervous. We all get anxious sometimes, but anxiety becomes a problem when it starts affecting your daily life drastically. The best thing that you can do to help reduce anxiety is to take a few deep breaths and talk to someone about what you're feeling. Some natural remedies that you can try include: getting enough sleep meditating, staying active and exercising, eating a healthy diet, staying active and working out, avoiding alcohol, avoiding caffeine, and quitting smoking cigarettes. Support your friends and loved ones by checking up on them and just listening. Whatever you're feeling is valid."
          ]
      },
      {
          "tag": "depression",
          "patterns": [
              "Depression"
              "I think I have depression",
              "What is depression?",
              "Tell me about depression",
              "I want to know about depression",
              "Can I have some information about depression?",
              "How do I support a loved one with depression?",
              "How do I fix my depression?"
          ],
          "responses": [
              "I see that you want to learn more about depression, and how to support yourself or your loved ones. At the higest level, depression is a mood disorder that causes a persistent feeling of sadness and loss of interest. Also called major depressive disorder or clinical depression, it affects how you feel, think and behave and can lead to a variety of emotional and physical problems. You may have trouble doing normal day-to-day activities, and sometimes you may feel as if life isn't worth living. All of us experience sadness and periods of time where things aren't that great. However, the difference in sadness and depression lies in the duration and magnitude of the problem. The best recommended course of treatment is to talk a therapist or psychiatrist who can help you find the best path for you. Everyone is different and what might work for one person might not work for another one. Some natural remedies that you can try include: getting enough sleep, meditating, staying active and exercising, eating a healthy diet, staying active and working out, avoiding alcohol, avoiding caffeine, and quitting smoking cigarettes. Support your friends and loved ones by checking up on them and just listening. Whatever you're feeling is valid."
          ]
      },
      {
          "tag": "schizophrenia",
          "patterns": [
              "schizophrenia"
              "I think I have schizophrenia",
              "What is schizophrenia?",
              "Tell me about schizophrenia",
              "I want to know about schizophrenia",
              "Can I have some information about schizophrenia?",
              "How do I support a loved one with schizophrenia?",
              "How do I fix my schizophrenia?"
          ],
          "responses": [
              "I see that you want to learn more about schizophrenia, and how to support yourself or your loved ones. At the higest level, Schizophrenia is a serious mental disorder in which people interpret reality abnormally. Schizophrenia may result in some combination of hallucinations, delusions, and extremely disordered thinking and behavior that impairs daily functioning, and can be disabling. People with schizophrenia require lifelong treatment. Early treatment may help get symptoms under control before serious complications develop and may help improve the long-term outlook. If you believe you have symptoms of schizophrenia, and/or you have various risk factors that increase the likelihood of schizophrenia, please consult a family doctor and/or psychiatrist for a diagnosis and treatment plan specific for the individual."
          ]
      },
      {
          "tag": "funny",
          "patterns": [
              "Tell me a joke!",
              "Tell me something funny!",
              "Do you know a joke?"
          ],
          "responses": [
              "What did the snail who was riding on the turtle's back say? Wheeeee!",
              "I was going to tell a time traveling joke, but you guys didn't like it.",
               "What do you call a lazy kangaroo? A pouch potato.",
              "I used to run a dating service for chickens, but I was struggling to make hens meet.",
              "Why do we tell actors to *break a leg?* Because every play has a cast.",
              "What does a pig put on dry skin? Oinkment.",
              "What do you call it when a snowman throws a tantrum? A meltdown.",
              "My uncle named his dogs Timex and Rolex. They're his watch dogs.",
              "Did you hear about the guy whose left side was cut off? He's all right now.",
              "How do you open a banana? With a mon-key.",
              "What did the buffalo say when his son left for college? Bison!"
          ]
      },
        {"tag": "bad",
         "patterns": ["I cant help but feel like a failure",
                      "I am sad",
                      "I can't sleep",
                      "trouble sleeping",
                      "I am struggling a lot in my daily life",
                      "I hate my life",
                      "I think I might be depressed",
                      "I am not doing so great",
                      "I am not good",
                      "I am not well",
                      "I have been feeling down lately",
                      "I do not feel good these days.",
                      "I just feel so stuck these days",
                      "I feel worthless, I do not know what to do",
                      "Sometimes I get sad for no particular reason, my energy just drops and I lose motivation to do anything"],
         "responses": ["I understand that, It it totally normal to feel low sometimes. Especially in the midst of a global pandemic, this last year has been really hard for everyone and you have had your own struggles. First thing i'd suggest is getting the sleep you need or it will impact how you think and feel . I'd also suggest you to look at your life and find out what is going well in your life and what you can be grateful for . I believe everyone has something in their life to be grateful for. You can figure it out with some help. I would suggest you to start a writing journal, it is really going to help you with understanding your own emotions.",
                       "You sound like you have so much on your mind ! I am relieved to hear that you are opening up about your struggles , but not being able to sleep , feeling worthless , and like you shouldn't be here are big issues that need addressing . Try to open up to your loved ones. Having someone to listen to you is a gift to yourself . You deserve the help of someone helping you change your feelings of worthlessness",
                       "You have not been able to do this alone, and your don't have to be alone!. It's time to bring a change in your life! You can do this ! best to you !"],
         "context_set": ""
        },
        {"tag": "extreme",
         "patterns": ["I cannot talk to anyone. I have attempted suicide before. I need help",
                      "I have suicidal thoughts and tendencies",
                      "I hate myself, I do not want to exist anymore.",
                      "I am depressed, I do not wish to live anymore",
                      "I have no one in my life that I can talk to, I cannot live like this",
                      "I feel worthless all the time. Sometimes I want to end it all",
                      "It feels like the world has come down crumbling around me. I feel helpless",
                      "I feel nothingness in my soul, I have no reason to live",
                      "Suicide",
                      "Everyday is the same, I feel like I cannot come out of this hole I am in. I want to end it all",
                      "I am not capable of love, I am severly depressed",
                      "Where else can I get help?"],
         "responses": ["Please understand that you are not alone and it is not too late. If you need immediate proffestional attention I am providing you with some resources, please connect with a professional Therapist." + "\n" + "Resources: 1. Suicide Helpline India: 9152987821" "\n"  "2. Help centers for different states: http://www.aasra.info/helpline.html" +"\n" + "3. Suicide prevention: https://www.who.int/health-topics/suicide#tab=tab_1", 
                       "Listen friend, whatever it is that you are going through, you can get help, you can take control of your life. In my opinion, it is time to reach out for help to a professional. I believe you are in need of proffesional care. Here are some resources to help you out" +"\n"+ "Resources: 1. Suicide Helpline India: 9152987821"+ "\n"+ "2. Help centers for different states: http://www.aasra.info/helpline.html" +"\n" +"3. Suicide prevention: https://www.who.int/health-topics/suicide#tab=tab_1"],
         "context_set": ""
        },
        {"tag": "good",
         "patterns": ["I feel happy!",
                      "I am better than before!",
                      "I am happy!",
                      "I am great",
                      "I feel great",
                      "I feel good these days",
                      "Today was a good day",
                      "I have found a new light in my life",
                      "I feel like i have a new purpose in life!"],
         "responses": ["Wow that is so great, I am happy our sessions have been working out for you!."],
         "context_set": ""
        },
        {"tag": "quotes",
         "patterns": ["Make me happy",
                      "Cheer me up",
                      "I need to change my mood",
                      "Tell me something",
                      "Tell me some good quotes",
                      "Need some positivity",
                      "Need to hear something good",
                      "I need to cheer up"],
         "responses": [" 'There is love, there is music, there is no limit, there is work, there is the precious sense that this is the hour of grace when all things gather and distil to create the rest of my life. I don’t believe in God, I believe in everything.' ",
                       " 'Tomorrow is always fresh, with no mistakes in it yet' ",
                       " 'Today will be better. Not because the world has dealt you better cards, but because you choose to use them to your advantage. Today, you choose happiness.' ",
                       " 'Happiness is not something ready made. It comes from your own actions.' ",
                       " 'Happiness can be found, even in the darkest of times, if only one remembers to turn on the light.' ",
                       " 'Imagine walking over a ray of sunshine. Everything is warm and full of light and there is a smile, not on your lips, but rather deep inside of you. There is no past or future, time doesn't exist. You feel like you are becoming one with the sun. You're walking through infinity, one step at a time. Turns out happiness was always a part of you, you just had to allow it to break through. And freedom, freedom was just one tiny step in front of you.' ",
                       " 'So far, you have made it through all of your worst days. Well done, Keep going friend!' ",
                       " 'to be kind to all, to like many and love a few, to be needed and wanted by those we love, is certainly the nearest we can come to happiness' ",
                       " 'If happiness visits you again, do not remember it's previous betrayal. Enter into the happiness and burst' ",
                       " 'I had the epiphany that laughter was light, and light was laughter, and that this was the secret of the universe.' "],
         "context_set": ""
        }
   ]
}

try:
    with open("data.pickle", "wb") as f:
        words, labels, training, output = pickle.load(f) #saving these variables in the file
except:
    #these blank lists are created as we want to go through the json file
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern) #tokenize the words, stemming
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag']) #gives what intent the tag is a part of
            
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    #removing all the duplicate elements
    words = [stemmer.stem(w.lower()) for w in words if w != "?"] #removing any question marks to not have any meaning to our model
    words = sorted(list(set(words)))

    labels = sorted(labels)
    #create training and testing output
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        #list to keep a check on what words are present
        #stemming the words
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        #going through the words and adding the information to bag
        for w in words:
            if w in wrds: #word exsits so add 1 to the list
                bag.append(1)
            else: #word does not exsit so add 0 to the list
                bag.append(0)
        print(docs_x)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
      
        training.append(bag)
        output.append(output_row)
        
training = numpy.array(training)
output = numpy.array(output)


with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)


try:
    x
    model.load("model.tflearn")
    model.summary()
except:
    #resetting the graph data
    tensorflow.compat.v1.reset_default_graph()
    #defines the input shape
    net = tflearn.input_data(shape=[None, len(training[0])])
    #8 neurons for the first hidden layer
    net = tflearn.fully_connected(net, 6)
    #8 neurons for the second hidden layer
    net = tflearn.fully_connected(net, 6)
    #gets probability for each neuron in the output layer,
    #the neuron which has the highest probability that word is our output
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    #we show the model the data 1000 times, the more times it sees the data, the more accurate it should get
    model.fit(training, output, n_epoch=300, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    #print(model.summary())

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))] #creates a blank bag list

    s_words = nltk.word_tokenize(s) #list of tokenized words
    s_words = [stemmer.stem(word.lower()) for word in s_words] #stemming the words

    for x in s_words:
        for i, w in enumerate(words):
            if w == x: #if current word is equal to our word in the sentence, then add 1 to bag list, generates the bag of words
                bag[i] = 1
    return numpy.array(bag)

bot_name = "HEAL"

def get_response(inp):
        if inp.lower() == "quit": #way to get out of the program
            return
        results = model.predict([bag_of_words(inp, words)]) #makes prediction, this only gives us some probability, no meaningful output
        results_index = numpy.argmax(results) #this gives the index of the greatest value in our list
        tag = labels[results_index] #maps the word to a particular tag
        #if results[results_index] > 0.6:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return random.choice(responses) #selects a response from the tag

    


#def chat():
    
    

#chat()





