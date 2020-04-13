#include <bits/stdc++.h>
#include <fstream>
#include <string>

using namespace std;

class CSVHelper {

    private:
        fstream fs;
        const char delim = ',';
        const char nl = '\n';

    public:
        vector<vector<float>> data;

        void openFile(string fileName) {
            fs.open(fileName, fstream::out);
        }

        void closeFile() {
            fs.close();
        }

        int isFileOpen() {
            return fs.is_open();
        }

        // stackoverflow.com/questions/1784573/iterator-for-2d-vector
        void writeOutput() {
            vector< vector<float> >::iterator row;
            vector<float>::iterator col;
            for (row = data.begin(); row != data.end(); row++) {
                for (col = row->begin(); col != row->end(); col++) {
                    fs.write((char *)&col,sizeof(float));
                    fs.write(&delim, sizeof(delim));
                }
                fs.write(&nl, sizeof(nl));
            }
        }
};

int main() {

    CSVHelper csv;

    csv.openFile("test.csv");



    return 0;
}

