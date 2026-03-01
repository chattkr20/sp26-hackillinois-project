# Catty Micro-Pillar

**Website**: [https://hackillinois26.milindkumar.dev](https://hackillinois26.milindkumar.dev)

## HackIllinois Info

**Project Track:** Caterpillar

**HackIllinois Prizes:** Most Creative, Best UI UX Design

**Company Prizes:** Best Use of OpenAI API


## Project Info

This web app allows a machine operator to create a status report for a machine, going through each part of the machine and providing a status by analyzing a recording of the operator speaking about the part and a recording of the part itself using AI.

Here's how the flow will work. When the user first begins using this web app, they fill out information about themselves (called Profile information). After that, every time the user wants to create a report about a machine, they press the microphone button and then can go hands-free until the report is created. The user then goes through each part of the machine and states the Part Name, records audio of the part (Machine Sound), and records themselves talking about the part (Description). The user uses voice commands to signal what they intend to record. After giving the phrase to signal submission, the Machine Sound is analyzed for discrepancies and concerning sounds. This analysis is combined with a transcription of the Description to create a final decision on each part consisting of the part's status, description, etc. These decisions about each part are then compiled into a report for the machine as a whole which the user can then view.

## Running the Project Locally

### Running Backend Locally

```
cd backend
```

#### Windows

```
./start.ps1
```

#### Mac / Linux

```
chmod +x start.sh
./start.sh
```

### Running Frontend Locally

```
cd frontend
npm i
npm run dev
```

It is recommended you use **Chrome** to run the localhost