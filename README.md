# Seoultech EE 2025 Capstone Design (이민형, 김수왕, 김민지)
### 이민형, 김수왕, 김민지 

# Project Description

**Smart Home Assitance System for the Hearing Impaired**

### 1. Introduction
1. Background and Motivation  

Hearing plays a crucial role in how people perceive and interact with their environment.
Through sound, individuals can:
	•	Determine the direction and distance of objects or events
	•	Understand the structure of a space
	•	Grasp the context of a situation

These auditory cues help people detect unseen dangers or recognize when someone is calling them.

However, individuals with hearing impairments face significant challenges in accessing this information.
Without auditory input, it becomes difficult to:
	•	Recognize spatial and situational cues
	•	Stay aware of potential hazards
	•	Respond effectively to surrounding events

As a result, they may experience daily inconveniences and safety risks, which can limit their independence in everyday life.

2. Project Objectives  

To develop an affordable home network-based system that collects sound data and analyzes it using deep learning,
with the goal of providing hearing-impaired individuals with real-time information about the location and context of acoustic events in their living environment.

3. System Overview  

The system consists of multiple sensor nodes installed in each room, which collect audio data in real-time.
These sound recordings are transmitted to a central home server, where deep learning-based inference is performed to classify the events.
After inference, a notification management algorithm filters out duplicate or unnecessary alerts.
Finally, only meaningful notifications—containing event type and location—are delivered to the user through the designated interface.

### 2. Related Work
1. Existing Sound Recognition Systems  
2. Accessibility Technologies in Smart Homes  
3. Comparison with Similar Research  

### 3. System Design
- 6월 4주차 개발 진행 예정
1. Hardware Architecture  
2. Hardware Configuration  
3. Central Hub and Server Design  
4. Communication Protocols  

### 4. Sound Classification
- 4월 4주차, 5월 1주차 개발 완료 
1. Dataset Collection and Preprocessing  
    - 개발 완료 
2. PANNs Model and Transfer Learning  
    - 개발 완료 
3. Class Definitions and Label Mapping
    - 개발 완료 
4. Embedding Extraction and Classifier Design  
    - 개발 완료 

### 5. Alert Management System
- 5월 2주차 개발 완료 
1. Mapping Sound Classes to Alert Priorities 
    - 개발 완료
2. Personalized Notifications (Vibration, Visuals, Logging)  
    - 개발 완료 

### 6. Implementation 
1. Data Flow and System Workflow 
    - 개발완료 
2. Integration of ESP32 and Server  
    - 현재 개발 중
3. Implementation Images and Descriptions  
    - 현재 개발 중 

### 7. Web Page 
1. Backend
    - 현재 개발 중 
    - Flask
2. DB system integration
    - 현재 개발 중 
3. Frontend
    - 현재 개발 중 

----------------------- 2025 Q3,Q4 --------------------------

### 8. Evaluation
1. Classification Accuracy and Model Performance  
    - 모의 테스트 진행 완료, 정확도 98% 
2. Real-World Testing Results  
3. User Feedback (Optional)  

### 9. Conclusion
1. Summary and Achievements  
2. Limitations and Challenges  
3. Future Work and Improvements  
