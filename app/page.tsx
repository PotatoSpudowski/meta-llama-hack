"use client"
import { useState } from 'react';
import Button from '@mui/material/Button';
import { Card, CardContent, Grid2, TextField } from '@mui/material';

export default function HorizontalLinearStepper() {
  // State variables for form inputs
  const [location, setLocation] = useState('');
  const [budget, setBudget] = useState('');
  const [ageGroupMin, setAgeGroupMin] = useState('');
  const [additionalInfo, setAdditionalInfo] = useState('');
  const [businessCategory, setBusinessCategory] = useState('');
  const [targetGender, setTargetGender] = useState('');

  const handleNext = () => {
    console.log({
      location,
      budget,
      ageGroupMin,
      additionalInfo,
      businessCategory,
      targetGender
    })

  }

  const handleChange = (setter) => (event) => {
    setter(event.target.value);
  };


  return (
    <Grid2 container>
      <Grid2 size={{ xs: 6 }} pt={4}>
        <Card>
          <CardContent>
            <Grid2 container mt={4}>
              <Grid2 size={{ xs: 4 }}>
                <TextField
                  required
                  id="location"
                  label="Location"
                  value={location}
                  onChange={handleChange(setLocation)}
                />
              </Grid2>
              <Grid2 size={{ xs: 4 }}>
                <TextField
                  required
                  id="budget"
                  label="Budget"
                  value={budget}
                  onChange={handleChange(setBudget)}
                />
              </Grid2>
              <Grid2 size={{ xs: 4 }}>
                <TextField
                  required
                  id="ageGroupMin"
                  label="Age Group min"
                  value={ageGroupMin}
                  onChange={handleChange(setAgeGroupMin)}
                />
              </Grid2>
              <Grid2 size={{ xs: 4 }} mt={2}>
                <TextField
                  required
                  id="additionalInfo"
                  label="Age Group max"
                  value={additionalInfo}
                  onChange={handleChange(setAdditionalInfo)}
                />
              </Grid2>
              <Grid2 size={{ xs: 4 }} mt={2}>
                <TextField
                  required
                  id="businessCategory"
                  label="Business Category"
                  value={businessCategory}
                  onChange={handleChange(setBusinessCategory)}
                />
              </Grid2>
              <Grid2 size={{ xs: 4 }} mt={2}>
                <TextField
                  required
                  id="targetGender"
                  label="Target Gender"
                  value={targetGender}
                  onChange={handleChange(setTargetGender)}
                />
              </Grid2>

              <Grid2 size={{ xs: 6 }} mt={2}>
                <input type="file" id="myFile" name="filename" />
              </Grid2>
              <Grid2 size={{ xs: 12 }} mt={2}>
                <Button onClick={() => handleNext()}>Submit</Button>
              </Grid2>

            </Grid2>
          </CardContent>
        </Card>
      </Grid2>
      <Grid2 size={{ xs: 6 }}>
        hey
      </Grid2>
    </Grid2>
  );
}