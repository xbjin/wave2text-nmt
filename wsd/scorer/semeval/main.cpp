#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

vector<string> split(string line)
{
    vector<string> tokens;
    istringstream line_stream(line);
    string token;
    while (line_stream >> token)
    {
        tokens.push_back(token);
    }
    return tokens;
}

string extract_lemma(string token)
{
    auto lemma_sense_delimiter_position = token.find_first_of('%');
    string lemma = token.substr(0, lemma_sense_delimiter_position);
    //cout << "lemma : " << lemma << endl;
    return lemma;
}

string extract_lemma(vector<string> tokens)
{
    if (tokens.empty()) return "";
    else return extract_lemma(tokens[0]);
}

string extract_sense(string token)
{
    auto lemma_sense_delimiter_position = token.find_first_of('%');
    string sense = "";
    if (lemma_sense_delimiter_position == token.npos)
    {
        sense = "";
    }
    else if (lemma_sense_delimiter_position + 1 == token.npos)
    {
        sense = "";
    }
    else
    {
        sense = token.substr(lemma_sense_delimiter_position + 1);
    }
    //cout << "sense : " << sense << endl;
    return sense;
}

vector<string> extract_senses(vector<string> tokens)
{
    vector<string> senses;
    for (string token : tokens)
    {
        senses.push_back(extract_sense(token));
    }
    return senses;
}

bool one_of_senses_match(string hyp_sense, vector<string> ref_senses)
{
    for (string ref_sense : ref_senses)
    {
        if (ref_sense == hyp_sense)
        {
            return true;
        }
    }
    return false;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage : <program> <text.tag> <answers.ans>" << endl;
        return 0;
    }

    string hyp_file_path = argv[1];
    string ans_file_path = argv[2];

    ifstream hyp_file(hyp_file_path);
    ifstream ans_file(ans_file_path);

    string hyp_word;
    string ans_word;

    double match = 0;
    double total = 0;

    string ans_line;
    string hyp_line;

    int window = 10;

    while (getline(ans_file, ans_line))
    {
        vector<string> ans_words = split(ans_line);
        string ans_lemma = extract_lemma(ans_words);
        vector<string> ans_senses = extract_senses(ans_words);

        int i = 0;
        bool found = false;
        auto position = hyp_file.tellg();
        while (!found && i < window && hyp_file >> hyp_word)
        {
            string hyp_lemma = extract_lemma(hyp_word);
            string hyp_sense = extract_sense(hyp_word);
            if (hyp_lemma == ans_lemma)
            {
                found = true;
                if (one_of_senses_match(hyp_sense, ans_senses))
                {
                    match += 1;
                }
            }
            i++;
        }
        if (!found)
        {
            hyp_file.seekg(position);
        }
        total += 1;
    }

    cout << (match / total) * 100.0 << endl;
    cout.flush();

    return 0;
}
